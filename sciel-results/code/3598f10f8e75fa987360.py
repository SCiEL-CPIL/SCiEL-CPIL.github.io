"""Optimize 8 planar points to maximize product of pairwise distances at fixed diameter."""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _score(points):
    """Compute Delta/n^n where Delta is product of distances over ordered pairs, after scaling to diameter 2."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    # Compute all pairwise distances
    # For ordered pairs: distance from i to j for all i != j
    # Product of all 56 distances
    # Use log to avoid underflow
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-15:
                return -np.inf
            log_prod += math.log(d)
    # Scale to diameter 2: find max distance
    max_d = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d > max_d:
                max_d = d
    if max_d < 1e-15:
        return -np.inf
    # After scaling by 2/max_d, each distance d becomes d * (2/max_d)
    # Product scales by (2/max_d)^(n*(n-1))
    # log_prod_scaled = log_prod + n*(n-1)*log(2/max_d)
    log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / max_d)
    # Score = Delta / n^n = exp(log_prod_scaled) / n^n
    # log_score = log_prod_scaled - n*log(n)
    log_score = log_prod_scaled - n * math.log(n)
    return math.exp(log_score) if log_score < 50 else float('inf')

def _neg_log_score(params):
    """Negative log of the score for minimization. Returns large value if invalid."""
    if time.time() > _DEADLINE:
        return 1e10
    n = 8
    pts = params.reshape(n, 2)
    # Check distinctness
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-10:
                return 1e10
    # Compute log product of ordered pair distances
    log_prod = 0.0
    max_d = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-15:
                return 1e10
            log_prod += math.log(d)
            if d > max_d:
                max_d = d
    if max_d < 1e-15:
        return 1e10
    log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / max_d)
    log_score = log_prod_scaled - n * math.log(n)
    return -log_score

def _regular_octagon():
    pts = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        pts.append([math.cos(angle), math.sin(angle)])
    return pts

def _two_concentric_squares(r1, r2, rot1, rot2):
    """Two squares, inner radius r1 rotated by rot1, outer radius r2 rotated by rot2."""
    pts = []
    for k in range(4):
        angle = rot1 + 2 * math.pi * k / 4
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
    for k in range(4):
        angle = rot2 + 2 * math.pi * k / 4
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle)])
    return pts

def _random_initial(scale=1.0):
    """Generate random points near the unit circle."""
    pts = []
    for k in range(8):
        angle = 2 * math.pi * k / 8 + (np.random.randn() * 0.3)
        r = 1.0 + np.random.randn() * 0.2
        pts.append([r * math.cos(angle), r * math.sin(angle)])
    return pts

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0
    
    best_points = _regular_octagon()
    best_score = _score(best_points)
    
    # List of initial configurations to try
    initial_configs = []
    
    # 1. Regular octagon
    initial_configs.append(_regular_octagon())
    
    # 2. Two concentric squares with various parameters
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [1.0, 1.2, 1.5]:
            for rot1 in [0.0, math.pi/8]:
                for rot2 in [0.0, math.pi/8]:
                    initial_configs.append(_two_concentric_squares(r1, r2, rot1, rot2))
    
    # 3. Random perturbations of the octagon
    np.random.seed(42)
    for _ in range(5):
        initial_configs.append(_random_initial())
    
    # 4. Square with 4 extra points
    for _ in range(3):
        pts = []
        # Square
        for k in range(4):
            angle = math.pi/4 + 2 * math.pi * k / 4
            pts.append([math.cos(angle), math.sin(angle)])
        # 4 more points at random positions
        for k in range(4):
            angle = np.random.uniform(0, 2*math.pi)
            r = np.random.uniform(0.3, 1.5)
            pts.append([r * math.cos(angle), r * math.sin(angle)])
        initial_configs.append(pts)
    
    # Try each initial configuration with Nelder-Mead
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > _DEADLINE:
            break
        
        init_params = np.array(init_pts, dtype=np.float64).flatten()
        
        try:
            result = minimize(
                _neg_log_score,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        if time.time() > _DEADLINE:
            break
        
        if result.success or result.fun < 1e8:
            final_pts = result.x.reshape(8, 2)
            # Check validity
            valid = True
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    d = math.sqrt((final_pts[i][0] - final_pts[j][0])**2 + 
                                  (final_pts[i][1] - final_pts[j][1])**2)
                    if d < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                sc = _score(final_pts)
                if sc > best_score:
                    best_score = sc
                    best_points = [list(p) for p in final_pts]
    
    # Try a few more random restarts if time permits
    np.random.seed(123)
    for _ in range(10):
        if time.time() > _DEADLINE:
            break
        
        init_pts = _random_initial()
        init_params = np.array(init_pts, dtype=np.float64).flatten()
        
        try:
            result = minimize(
                _neg_log_score,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        if time.time() > _DEADLINE:
            break
        
        if result.success or result.fun < 1e8:
            final_pts = result.x.reshape(8, 2)
            valid = True
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    d = math.sqrt((final_pts[i][0] - final_pts[j][0])**2 + 
                                  (final_pts[i][1] - final_pts[j][1])**2)
                    if d < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                sc = _score(final_pts)
                if sc > best_score:
                    best_score = sc
                    best_points = [list(p) for p in final_pts]
    
    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, p[0]))
        y = max(-1e6, min(1e6, p[1]))
        final_result.append([float(x), float(y)])
    
    return final_result
