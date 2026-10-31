import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.special import logsumexp

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _smooth_min_area(points, beta):
    """
    Smooth approximation of min area using log-sum-exp.
    Returns a differentiable proxy for min area.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Smooth min via log-sum-exp
    # min(x) ≈ -logsumexp(-x) / beta for large beta
    # More precisely: min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    # = -(1/beta) * logsumexp(-beta * x)
    smooth_min = -logsumexp(-beta * areas) / beta
    
    return smooth_min

def _objective(points, beta):
    """
    Objective to maximize: smooth_min_area - penalty
    Returns negative for minimization algorithms.
    """
    sm = _smooth_min_area(points, beta)
    pen = _penalty(points)
    return -(sm - 1000.0 * pen)

def _polish(points, beta=50.0, maxiter=200, time_limit=10.0):
    """
    Polish a configuration using L-BFGS-B with smooth objective.
    """
    start_time = time.time()
    
    def _obj(flat_pts):
        pts = flat_pts.reshape(-1, 2)
        return _objective(pts, beta)
    
    try:
        res = minimize(
            _obj,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8}
        )
        polished = res.x.reshape(-1, 2)
        return polished
    except Exception:
        return points

def _simulated_annealing(points, rng, start_temp=0.05, end_temp=0.001,
                          n_iter=5000, beta_schedule=None, time_limit=30.0):
    """
    Simulated annealing with adaptive step size and annealed log-sum-exp polishing.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_area = _min_area(points)
    current_pts = points.copy()
    current_area = best_area
    current_temp = start_temp
    
    for i in range(n_iter):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size: larger early, smaller later
        progress = i / n_iter
        step_size = start_temp * (1 - progress) + end_temp * progress
        
        # Generate candidate
        candidate = current_pts.copy()
        idx = rng.integers(0, N)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        candidate[idx, 0] += dx
        candidate[idx, 1] += dy
        
        # Check if inside triangle (rough check)
        x, y = candidate[idx, 0], candidate[idx, 1]
        if y < -1e-10 or y > S3 * x + 1e-10 or y > S3 * (1 - x) + 1e-10:
            continue
        
        # Evaluate
        cand_area = _min_area(candidate)
        
        # Accept if better or with probability exp((cand_area - current_area) / current_temp)
        if cand_area > current_area:
            current_pts = candidate
            current_area = cand_area
            if cand_area > best_area:
                best_area = cand_area
                best_pts = candidate.copy()
        else:
            delta = cand_area - current_area
            if delta < 0 and current_temp > 1e-10:
                prob = np.exp(delta / current_temp)
                if rng.random() < prob:
                    current_pts = candidate
                    current_area = cand_area
        
        # Periodic polishing with smooth objective
        if i % 500 == 499:
            polished = _polish(current_pts, beta=50.0, maxiter=50, time_limit=2.0)
            pol_area = _min_area(polished)
            if pol_area > current_area:
                current_pts = polished
                current_area = pol_area
                if pol_area > best_area:
                    best_area = pol_area
                    best_pts = polished.copy()
        
        # Anneal temperature
        current_temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
    
    return best_pts, best_area

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = -np.inf
    
    # Phase 1: Multiple random restarts with SA + polish
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        
        # Random initial configuration
        init_pts = _initial_population(rng, 1)[0]
        
        # Quick polish to get a decent starting point
        init_pts = _polish(init_pts, beta=50.0, maxiter=100, time_limit=5.0)
        init_area = _min_area(init_pts)
        
        # SA from this point
        sa_pts, sa_area = _simulated_annealing(
            init_pts, rng,
            start_temp=0.03, end_temp=0.0005,
            n_iter=3000, time_limit=8.0
        )
        
        # Final polish
        final_pts = _polish(sa_pts, beta=100.0, maxiter=200, time_limit=5.0)
        final_area = _min_area(final_pts)
        
        if final_area > best_area:
            best_area = final_area
            best_pts = final_pts
    
    # Phase 2: If we have time, do more restarts from best with perturbations
    if best_pts is not None and time.time() - start_time < timeout - 15:
        for _ in range(5):
            if time.time() - start_time > timeout - 5:
                break
            
            # Perturb the best point
            pts = best_pts.copy()
            for i in range(N):
                pts[i, 0] += rng.normal(0, 0.02)
                pts[i, 1] += rng.normal(0, 0.02)
            
            # SA from perturbed point
            sa_pts, sa_area = _simulated_annealing(
                pts, rng,
                start_temp=0.02, end_temp=0.0005,
                n_iter=2000, time_limit=5.0
            )
            
            # Polish
            final_pts = _polish(sa_pts, beta=100.0, maxiter=150, time_limit=3.0)
            final_area = _min_area(final_pts)
            
            if final_area > best_area:
                best_area = final_area
                best_pts = final_pts
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() - start_time < timeout - 3:
        best_pts = _polish(best_pts, beta=200.0, maxiter=300, time_limit=10.0)
        best_area = _min_area(best_pts)
    
    # Ensure points are inside the triangle
    if best_pts is not None:
        x = best_pts[:, 0]
        y = best_pts[:, 1]
        
        # Project to triangle
        for _ in range(5):
            y = np.maximum(y, 0)
            viol1 = y > S3 * x
            if np.any(viol1):
                x[viol1] = y[viol1] / S3
            viol2 = y > S3 * (1 - x)
            if np.any(viol2):
                x[viol2] = 1 - y[viol2] / S3
            y = np.maximum(y, 0)
        
        best_pts[:, 0] = x
        best_pts[:, 1] = y
        
        # Final area check after projection
        final_area = _min_area(best_pts)
        if final_area > best_area:
            best_area = final_area
    
    if best_pts is None:
        # Fallback: return a random valid configuration
        best_pts = _initial_population(rng, 1)[0]
    
    return best_pts

construct()
