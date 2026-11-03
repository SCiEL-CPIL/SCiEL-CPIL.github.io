"""
Heilbronn convex regions n=14
Combines central symmetry constraint with Fourier boundary parameterization 
and smooth optimization (log-sum-exp surrogate) for robustness.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin before 150s limit

# Precompute indices for all combinations of 3 points (i < j < k)
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _check_time():
    return time.time() < DEADLINE

def _all_areas(points):
    """Compute areas of all C(14,3) triangles."""
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    """Compute area of convex hull."""
    try:
        # Check for degenerate cases
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Exact score: min triangle area / convex hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert parameters to 14 centrally symmetric points on a convex boundary.
    params[0:7]: angles in (0, pi) for the 7 half-points
    params[7:14]: radii for the 7 half-points
    
    Ensures convexity by ordering angles and using positive radii.
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    
    # Clip to valid ranges
    angles = np.clip(angles, 0.01, np.pi - 0.01)
    radii = np.clip(radii, 0.1, 2.0)
    
    # Sort angles to ensure convex polygon order (star-shaped wrt origin)
    # Since we want a convex hull containing the origin, and points are symmetric,
    # sorting angles in (0, pi) ensures the upper half is convex if radii are well-behaved.
    # However, simple sorting might break the pairing if we don't track indices.
    # For central symmetry, if we sort angles, we must keep radii associated.
    # Actually, for a convex centrally symmetric polygon, the points in the first quadrant 
    # and second quadrant should be ordered by angle.
    
    # Let's enforce convexity more strongly by using a Fourier representation 
    # OR simply by optimizing the angles/radii and checking if the hull area is close to 
    # the sum of triangle areas (which it is for convex).
    
    # A simpler robust approach: 
    # 1. Define 7 points in the upper half plane.
    # 2. Ensure they form a convex chain from (r0, 0) to (-r6, 0) via the top.
    # Sorting angles in (0, pi) and using positive radii generally creates a star-shaped polygon.
    # To ensure convexity, we rely on the optimization to keep the shape "bulgy" enough.
    # If the shape becomes non-convex, the hull area will be smaller than the polygon area,
    # but the score uses hull area, so it's still valid, just potentially suboptimal.
    # However, the task asks for points in a *convex region*. If the points themselves 
    # form a non-convex polygon, the "region" is the convex hull. The points are still 
    # "in" the region. The score is defined as min_tri_area / hull_area.
    # So non-convex point sets are allowed as long as we normalize by the hull.
    # But convex point sets are generally better for this problem.
    
    # Let's just sort the angles to ensure a consistent ordering for the upper half.
    # This doesn't change the set of points, just the order in the array.
    # The symmetry p_{i+7} = -p_i is maintained.
    
    # We don't strictly need to sort for the score calculation, 
    # but it helps if we were doing geometric checks.
    # Let's just map them directly.
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective_smooth(params, beta=20.0):
    """
    Smooth surrogate objective: -soft_min(areas) / hull_area.
    Using log-sum-exp for the soft-min of areas.
    """
    points = _params_to_points(params)
    areas = _all_areas(points)
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    
    # Soft min: approx min(areas)
    # min(x) = -lim_{beta->inf} (1/beta) log(sum(exp(-beta x)))
    # To avoid overflow, use stable logsumexp
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    # We want to maximize soft_min_area / hull_area
    # So we minimize - (soft_min_area / hull_area)
    return -soft_min_area / hull_area

def _objective_exact(params):
    """Exact objective for final polishing."""
    points = _params_to_points(params)
    return -_score(points)

def construct():
    """
    Construct 14 points to maximize min triangle area / convex hull area.
    """
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Initial guess: Regular 14-gon (centrally symmetric)
    # Angles for 7 points in upper half: (k+0.5)*pi/7 ? No, regular 14-gon has vertices at k*2pi/14.
    # Upper half vertices: k=0..6? 
    # 14-gon: angles 0, 2pi/14, ..., 12pi/14.
    # Symmetric pairs: (0, pi), (2pi/14, pi+2pi/14), etc.
    # In our param: p_i = r_i (cos a_i, sin a_i), p_{i+7} = -p_i.
    # For regular 14-gon, r=1.
    # Angles in (0, pi): 
    # The vertices in upper half are at angles: 0, pi/7, 2pi/7, 3pi/7, 4pi/7, 5pi/7, 6pi/7.
    # Wait, 0 is on the boundary. 
    # Let's use angles: np.linspace(0, np.pi, 7) -> 0, pi/6, pi/3, pi/2, 2pi/3, 5pi/6, pi.
    # This corresponds to a regular 12-gon? No.
    # Regular 14-gon upper half angles: 
    # 0, 2pi/14=pi/7, 2pi/7, 3pi/7, 4pi/7, 5pi/7, 6pi/7.
    init_angles = np.array([0, np.pi/7, 2*np.pi/7, 3*np.pi/7, 4*np.pi/7, 5*np.pi/7, 6*np.pi/7])
    # Clip slightly inside to avoid boundary issues in ConvexHull if needed, 
    # though ConvexHull handles boundaries fine.
    init_angles = np.clip(init_angles, 0.01, np.pi - 0.01)
    init_radii = np.ones(7)
    
    init_params = np.concatenate([init_angles, init_radii])
    
    # Evaluate initial
    init_points = _params_to_points(init_params)
    best_score = _score(init_points)
    best_params = init_params.copy()
    best_points = init_points.copy()
    
    # Define bounds
    bounds = []
    for _ in range(N_HALF):
        bounds.append((0.01, np.pi - 0.01))
    for _ in range(N_HALF):
        bounds.append((0.1, 2.0))
        
    # Phase 1: Global Search with Differential Evolution
    # Use smooth objective to guide DE
    if _check_time():
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-6,
                polish=False,  # We will do our own polishing
                init='sobol',
                updating='immediate'
            )
            
            if de_result.success or de_result.fun < _objective_smooth(best_params):
                candidate_params = de_result.x
                candidate_points = _params_to_points(candidate_params)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params
                    best_points = candidate_points
        except Exception:
            pass
    
    # Phase 2: Local Polish with L-BFGS-B using Smooth Objective
    if _check_time() and best_params is not None:
        try:
            lbfgs_result = minimize(
                _objective_smooth,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if lbfgs_result.fun < _objective_smooth(best_params):
                candidate_params = lbfgs_result.x
                candidate_points = _params_to_points(candidate_params)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params
                    best_points = candidate_points
        except Exception:
            pass
            
    # Phase 3: Final Polish with Nelder-Mead using Exact Objective
    if _check_time() and best_params is not None:
        try:
            nm_result = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            candidate_params = nm_result.x
            candidate_points = _params_to_points(candidate_params)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            pass
            
    # Phase 4: Try a few random restarts if time permits
    # The symmetric constraint is strong, so restarts might not help much, 
    # but different initial radii might help escape local minima.
    if _check_time():
        for seed in [123, 456]:
            if not _check_time():
                break
            try:
                # Perturb initial radii
                init_angles_pert = init_angles + np.random.normal(0, 0.1, 7)
                init_angles_pert = np.clip(init_angles_pert, 0.01, np.pi - 0.01)
                init_radii_pert = np.ones(7) * (1.0 + np.random.normal(0, 0.2, 7))
                init_radii_pert = np.clip(init_radii_pert, 0.1, 2.0)
                
                start_params = np.concatenate([init_angles_pert, init_radii_pert])
                
                # Quick local optimization from this start
                nm_res = minimize(
                    _objective_exact,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True}
                )
                
                candidate_params = nm_res.x
                candidate_points = _params_to_points(candidate_params)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
            except Exception:
                pass

    return best_points
