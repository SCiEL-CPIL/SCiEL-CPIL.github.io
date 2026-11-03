import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
DEADLINE = 138.0  # Safety margin before 150s limit

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRIS = len(IDX_I)

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _soft_min_value(areas, beta):
    """Stable log-sum-exp based soft minimum."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _sym_params_to_points(params):
    """
    Convert 14 parameters (7 x, 7 y) to 14 points.
    Points are centrally symmetric: P_i and -P_i.
    We center the points to ensure symmetry around origin.
    """
    half_pts = params.reshape(N_HALF, 2)
    # Center the half points to ensure the full set is centered?
    # Actually, if we define P_i and -P_i, the centroid is 0.
    # But we might want to allow the "center" to shift? No, symmetry is defined w.r.t centroid.
    # So we just take the 7 points and their negatives.
    points = np.vstack([half_pts, -half_pts])
    return points

def _objective_sym_hard(params):
    """Hard objective for symmetric 14 points."""
    if np.any(~np.isfinite(params)):
        return 1.0
    pts = _sym_params_to_points(params)
    # Check for degeneracy
    if np.min(np.linalg.norm(pts, axis=1)) < 1e-9:
        return 1.0
    s = _score(pts)
    return -s if s > 0 else 1.0

def _objective_sym_soft(params, beta=100.0):
    """Soft objective for symmetric 14 points."""
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _sym_params_to_points(params)
    if np.min(np.linalg.norm(pts, axis=1)) < 1e-9:
        return 1e10
    
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    sm = _soft_min_value(areas, beta)
    if sm < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / sm

def _generate_sym_inits():
    """Generate diverse initial symmetric points."""
    inits = []
    
    # 1. Regular 14-gon symmetric subset
    # Angles for 7 points in [0, pi)
    angles = np.linspace(0, np.pi, N_HALF, endpoint=False)
    radii = np.ones(N_HALF)
    pts_half = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    inits.append(pts_half.flatten())
    
    # 2. Perturbed regular
    for seed in range(10):
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, np.pi, N_HALF, endpoint=False) + rng.normal(0, 0.1, N_HALF)
        radii = 1.0 + rng.normal(0, 0.1, N_HALF)
        radii = np.maximum(radii, 0.5)
        pts_half = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts_half.flatten())
        
    # 3. Two rings (symmetric)
    # 4 outer, 3 inner? Or 7 points split.
    # Let's just use random polar inits for the half
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        angles = np.sort(rng.uniform(0, np.pi, N_HALF))
        radii = 0.8 + rng.uniform(0, 0.2, N_HALF)
        pts_half = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts_half.flatten())
        
    return inits

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def remaining_time():
        return DEADLINE - (time.time() - start_time)

    # === PHASE 1: Differential Evolution on Symmetric Params ===
    # 14 parameters
    bounds = [(-1.0, 1.0)] * (2 * N_HALF)
    
    if remaining_time() > 60.0:
        try:
            res_de = differential_evolution(
                _objective_sym_hard,
                bounds,
                seed=42,
                maxiter=150,
                popsize=20,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            if res_de.fun < 0:
                pts14 = _sym_params_to_points(res_de.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    if remaining_time() > 40.0:
        try:
            res_de2 = differential_evolution(
                _objective_sym_hard,
                bounds,
                seed=123,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 0.8),
                recombination=0.85,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if res_de2.fun < 0:
                pts14 = _sym_params_to_points(res_de2.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # === PHASE 2: Local Refinement on Symmetric Params ===
    inits = _generate_sym_inits()
    
    # Add best from DE if available
    if best_points is not None:
        # We need to extract the symmetric params from best_points?
        # best_points is 14 points. We can take the first 7?
        # It's safer to just add the DE result directly if we stored it.
        # But we didn't store the params, just points.
        # Let's just rely on the generated inits and DE results.
        pass

    for i, init_params in enumerate(inits):
        if remaining_time() < 15.0:
            break
        
        # Stage A: L-BFGS-B with soft objective
        if remaining_time() > 12.0:
            try:
                res_lbfgs = minimize(
                    lambda x: _objective_sym_soft(x, beta=100.0),
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_lbfgs.fun < 1e10:
                    pts14 = _sym_params_to_points(res_lbfgs.x)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
                    init_params = res_lbfgs.x
            except Exception:
                pass

        # Stage B: Nelder-Mead on hard objective
        if remaining_time() > 8.0:
            try:
                res_nm = minimize(
                    _objective_sym_hard,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.fun < 0:
                    pts14 = _sym_params_to_points(res_nm.x)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage C: Powell on hard objective
        if remaining_time() > 6.0:
            try:
                res_pow = minimize(
                    _objective_sym_hard,
                    init_params,
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_pow.fun < 0:
                    pts14 = _sym_params_to_points(res_pow.x)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
