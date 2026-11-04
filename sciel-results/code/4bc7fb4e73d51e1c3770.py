"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for vectorization
_triple_indices = np.array(list(combinations(range(N), 3)))
_i_idx = _triple_indices[:, 0]
_j_idx = _triple_indices[:, 1]
_k_idx = _triple_indices[:, 2]

def _triangle_areas(pts):
    """Compute all triangle areas efficiently."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _exact_score(pts):
    """Compute the exact score: min triangle area / hull area."""
    hull_area = _hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min(areas, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    if len(areas) == 0:
        return 0.0
    min_r = np.min(areas)
    # Shift to avoid overflow: exp(-tau * (x - min_r)) <= 1
    shifted = -tau * (areas - min_r)
    # Clip to avoid underflow issues in sum, though log-sum-exp handles it well
    # max shifted is 0, min is very negative
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _params_to_points(params):
    """
    Convert 22 parameters to 13 points.
    p[0] = (0, 0), p[1] = (1, 0)
    p[2:] = params.reshape(11, 2)
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _points_to_params(pts):
    """
    Convert 13 points to 22 parameters by fixing p[0] and p[1].
    """
    # Translate so p[0] is at origin
    pts = pts - pts[0]
    # Rotate and scale so p[1] is at (1, 0)
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts[2:].flatten()

def _objective(params, tau):
    """
    Objective function for minimization.
    Returns -soft_min(areas / hull_area, tau).
    """
    pts = _params_to_points(params)
    hull_area = _hull_area(pts)
    if hull_area < 1e-9:
        return 1e9
    
    areas = _triangle_areas(pts)
    ratios = areas / hull_area
    approx_min = _soft_min(ratios, tau)
    return -approx_min

def _generate_initial_points(rng, method='random'):
    """Generate diverse initial point configurations."""
    if method == 'polygon':
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        # Perturb
        pts += rng.normal(0, 0.1, (N, 2))
        
    elif method == 'concentric':
        # 7 outer, 6 inner
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer_pts = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
        inner_pts = 0.6 * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
        pts = np.vstack([outer_pts, inner_pts])
        pts += rng.normal(0, 0.05, (N, 2))
        
    elif method == 'grid':
        # Approximate grid
        xs = np.linspace(0, 1, 4)
        ys = np.linspace(0, 1, 3)
        grid_pts = np.array([[x, y] for x in xs for y in ys]) # 12 points
        # Add center
        center = np.array([[0.5, 0.5]])
        pts = np.vstack([grid_pts, center])
        pts += rng.normal(0, 0.05, (N, 2))
        
    else:
        # Random
        pts = rng.uniform(-1, 1, (N, 2))
        
    # Ensure convexity is not strictly required for initialization, 
    # but we should avoid degenerate cases (collinear points)
    params = _points_to_params(pts)
    if params is None:
        params = rng.uniform(-1, 1, 2 * (N - 2))
    return params

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    best_pts = None
    
    # Bounds for the 22 free coordinates
    # We want to keep points within a reasonable range relative to the fixed edge (0,0)-(1,0)
    # A square of side 2 centered at (0.5, 0.5) might be too small if hull is large, 
    # but since we normalize by hull area, scale doesn't matter as much as relative positions.
    # However, if points are too far, hull area grows, but triangle areas also grow.
    # The ratio is scale-invariant.
    # Let's use a generous box, say [-3, 3]
    bounds = [(-3.0, 3.0)] * (2 * (N - 2))
    
    # Phase 1: Global Search with Differential Evolution
    # Use a moderate tau for DE to have a smoother landscape
    tau_de = 50.0
    de_time_budget = 40.0
    
    if (time.time() - start_time) < time_limit - de_time_budget - 20.0:
        try:
            de_start = time.time()
            de_res = differential_evolution(
                _objective,
                bounds,
                args=(tau_de,),
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                init='sobol',
                callback=lambda xk, conv: (time.time() - de_start) > de_time_budget
            )
            
            if de_res.x is not None:
                pts = _params_to_points(de_res.x)
                score = _exact_score(pts)
                if score > best_score:
                    best_score = score
                    best_params = de_res.x.copy()
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local Refinement with Annealing
    # Generate a set of starting candidates
    candidates = []
    if best_params is not None:
        candidates.append(best_params.copy())
    
    # Add diverse starts
    methods = ['polygon', 'concentric', 'grid', 'random', 'random']
    for m in methods:
        if (time.time() - start_time) > time_limit - 10.0:
            break
        p = _generate_initial_points(rng, m)
        candidates.append(p)
    
    # Annealing schedule for tau
    # Start smooth, end sharp
    tau_schedule = [50, 200, 1000, 5000, 20000]
    
    for idx, params in enumerate(candidates):
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        current_params = params.copy()
        
        for tau in tau_schedule:
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    current_params,
                    args=(tau,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfun': 5000, 'disp': False}
                )
                current_params = res.x
                
                # Evaluate exact score
                pts = _params_to_points(current_params)
                score = _exact_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                    best_pts = pts.copy()
                    
            except Exception:
                pass
    
    # Phase 3: Final Polish with Nelder-Mead on exact objective
    if best_params is not None and (time.time() - start_time) < time_limit - 2.0:
        try:
            def exact_obj(params):
                pts = _params_to_points(params)
                return -_exact_score(pts)
            
            nm_res = minimize(
                exact_obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            
            pts = _params_to_points(nm_res.x)
            score = _exact_score(pts)
            
            if score > best_score:
                best_score = score
                best_params = nm_res.x.copy()
                best_pts = pts.copy()
                
        except Exception:
            pass
    
    # Return the best points found
    if best_pts is None:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts
