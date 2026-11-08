"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _triangle_areas(pts):
    """Compute all triangle areas for N points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # For 2D, this is the area
    except Exception:
        return 1e-10

def _objective(params, beta):
    """Objective: minimize log-sum-exp of -beta*area_t / beta, normalized by hull area."""
    n = N
    pts = np.array(params).reshape(n, 2)
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    # log-sum-exp of -beta * area_t
    max_neg = np.max(-beta * areas)
    lse = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    obj = lse / beta / hull_area
    return obj

def _score(pts):
    """Compute the actual score: min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return np.min(areas) / hull_area

def _make_initial_params(seed):
    """Create initial parameterization with two-ring structure."""
    rng = np.random.RandomState(seed)
    params = np.zeros((N, 2))
    
    # Center point
    params[0] = [0.0, 0.0]
    
    # Ring 1: 5 points
    r1 = 0.5 + rng.uniform(-0.1, 0.1)
    angle_offset1 = rng.uniform(0, 2*np.pi)
    for i in range(5):
        angle = angle_offset1 + 2*np.pi*i/5
        params[i+1] = [r1*np.cos(angle), r1*np.sin(angle)]
    
    # Ring 2: 8 points
    r2 = 1.0 + rng.uniform(-0.1, 0.1)
    angle_offset2 = rng.uniform(0, 2*np.pi)
    for i in range(8):
        angle = angle_offset2 + 2*np.pi*i/8
        params[i+9] = [r2*np.cos(angle), r2*np.sin(angle)]
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple initializations with increasing beta
    n_starts = 8
    
    for start_idx in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        params0 = _make_initial_params(seed=42 + start_idx*7)
        
        # Use a moderate beta for L-BFGS-B
        beta = 50.0
        
        try:
            res = minimize(
                _objective,
                params0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success or res.fun < np.inf:
                pts = res.x.reshape(N, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_params = res.x.copy()
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
    
    # If we have time, do a final polish with higher beta
    if best_params is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            beta_high = 200.0
            try:
                res = minimize(
                    _objective,
                    best_params,
                    args=(beta_high,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < np.inf:
                    pts = res.x.reshape(N, 2)
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback: return a reasonable default
        return np.array(_make_initial_params(42)).reshape(N, 2)
    
    return best_params.reshape(N, 2)
# EVOLVE-BLOCK-END
