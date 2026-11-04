"""Heilbronn convex regions: maximize minimum triangle area over convex hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas from n points."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def _params_to_points(params):
    """Convert (angles, radii) params to 2D points.
    params: array of shape (2*N,) where first N are angles, next N are radii.
    Angles are sorted to maintain convex ordering."""
    angles = np.sort(params[:N])
    radii = params[N:]
    # Normalize angles to [0, 2*pi)
    angles = angles - angles[0]
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _soft_min_from_params(params, temperature=0.01):
    """Soft-min objective in parameter space."""
    pts = _params_to_points(params)
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    normalized = areas / hull_area
    shifted_neg = -normalized / temperature
    shifted_neg -= np.max(shifted_neg)
    log_sum = np.log(np.sum(np.exp(shifted_neg))) + np.max(shifted_neg)
    return temperature * log_sum

def _exact_objective_from_params(params):
    """Exact negative score in parameter space."""
    pts = _params_to_points(params)
    sc = _score(pts)
    return -sc

def _initial_params():
    """Create initial parameters for regular polygon."""
    angles = 2 * np.pi * np.arange(N) / N
    radii = np.ones(N)
    return np.concatenate([angles, radii])

def _perturb_params(params, rng, scale=0.1):
    """Perturb parameters."""
    new_params = params.copy()
    # Perturb angles
    angle_pert = rng.normal(0, scale, N)
    new_angles = params[:N] + angle_pert
    new_angles = np.sort(new_angles)
    # Perturb radii
    radii_pert = rng.normal(0, scale * 0.5, N)
    new_radii = np.clip(params[N:] + radii_pert, 0.3, 2.0)
    return np.concatenate([new_angles, new_radii])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_pts = None
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Start with regular polygon
    params0 = _initial_params()
    pts0 = _params_to_points(params0)
    score0 = _score(pts0)
    if score0 > best_score:
        best_score = score0
        best_pts = pts0.copy()
        best_params = params0.copy()
    
    # Phase 1: L-BFGS-B optimization in parameter space from multiple starts
    max_phase1_attempts = 15
    for attempt in range(max_phase1_attempts):
        if time.time() > deadline - 30.0:
            break
        
        if attempt == 0:
            params_init = params0.copy()
        else:
            params_init = _perturb_params(best_params, rng, scale=0.15)
        
        try:
            result = minimize(
                _soft_min_from_params,
                params_init,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                timeout=20
            )
            
            if time.time() > deadline - 30.0:
                break
            
            if result.fun < 1e6:
                params_opt = result.x
                pts_opt = _params_to_points(params_opt)
                if np.all(np.abs(pts_opt) < 10):
                    sc = _score(pts_opt)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts_opt.copy()
                        best_params = params_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated annealing in parameter space
    if best_params is not None and time.time() < deadline - 20.0:
        current_params = best_params.copy()
        current_score = best_score
        T = 0.1
        cooling = 0.95
        max_sa_iters = 2000
        
        for it in range(max_sa_iters):
            if time.time() > deadline - 15.0:
                break
            
            new_params = _perturb_params(current_params, rng, scale=0.05)
            new_pts = _params_to_points(new_params)
            if np.any(np.abs(new_pts) > 10):
                continue
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = new_pts.copy()
                    best_params = new_params.copy()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0 and rng.random() < np.exp(delta / T):
                    current_params = new_params
                    current_score = new_score
            
            T *= cooling
            if T < 1e-6:
                break
    
    # Phase 3: Final L-BFGS-B refinement on best solution
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            # Refine in 2D space
            result = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if time.time() < deadline - 5.0:
                pts_refined = result.x.reshape(N, 2)
                if np.all(np.abs(pts_refined) < 10):
                    sc = _score(pts_refined)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts_refined.copy()
        except Exception:
            pass
    
    if best_pts is None:
        return _params_to_points(params0)
    
    return best_pts
# EVOLVE-BLOCK-END
