"""Centrally symmetric polygon optimization for Heilbronn n=14 with soft-min refinement."""
N = 14
M = N // 2  # 7 unique points

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# Precompute indices for all combinations of 3 points for vectorized area computation
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    """Vectorized computation of all triangle areas."""
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
    """Compute convex hull area."""
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Min triangle area / convex hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _build_symmetric_points(unique_pts):
    """Build 14 points from 7 unique points and their negatives."""
    return np.vstack([unique_pts, -unique_pts])

def _objective_sym_hard(flat):
    """Hard objective for centrally symmetric parameterization."""
    if np.any(~np.isfinite(flat)):
        return 1e10
    pts_half = flat.reshape(M, 2)
    if np.any(np.linalg.norm(pts_half, axis=1) < 1e-6):
        return 1e10
    pts = _build_symmetric_points(pts_half)
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _objective_sym_soft(flat, beta=50.0):
    """Soft-min objective for centrally symmetric parameterization."""
    if np.any(~np.isfinite(flat)):
        return 1e10
    pts_half = flat.reshape(M, 2)
    if np.any(np.linalg.norm(pts_half, axis=1) < 1e-6):
        return 1e10
    pts = _build_symmetric_points(pts_half)
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_sym_full(flat):
    """Hard objective for full 28D parameterization."""
    if np.any(~np.isfinite(flat)):
        return 1e10
    pts = flat.reshape(N, 2)
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _objective_full_soft(flat, beta=50.0):
    """Soft-min objective for full 28D parameterization."""
    if np.any(~np.isfinite(flat)):
        return 1e10
    pts = flat.reshape(N, 2)
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _refine_sym_lbfgs(flat_init, deadline, beta=50.0):
    """Refine in 14D symmetric space using L-BFGS-B on soft-min objective."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        res = minimize(
            lambda x: _objective_sym_soft(x, beta=beta),
            flat_init,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 100000}
        )
        if time.time() > deadline:
            return None, -np.inf
        cand_pts = _build_symmetric_points(res.x.reshape(M, 2))
        cand_score = _score(cand_pts)
        return cand_pts, cand_score
    except Exception:
        return None, -np.inf

def _refine_sym_nm(flat_init, deadline):
    """Refine in 14D symmetric space using Nelder-Mead on hard objective."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        def obj(flat):
            if time.time() > deadline:
                return 1e10
            return _objective_sym_hard(flat)
        res = minimize(
            obj, flat_init, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if time.time() > deadline:
            return None, -np.inf
        if res.fun < 1e9:
            cand_pts = _build_symmetric_points(res.x.reshape(M, 2))
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
    except Exception:
        pass
    return None, -np.inf

def _refine_full_lbfgs(flat_init, deadline, beta=50.0):
    """Refine in 28D full space using L-BFGS-B on soft-min objective."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        res = minimize(
            lambda x: _objective_full_soft(x, beta=beta),
            flat_init,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 100000}
        )
        if time.time() > deadline:
            return None, -np.inf
        cand_pts = res.x.reshape(N, 2)
        cand_score = _score(cand_pts)
        return cand_pts, cand_score
    except Exception:
        return None, -np.inf

def _refine_full_nm(flat_init, deadline):
    """Refine in 28D full space using Nelder-Mead on hard objective."""
    if time.time() > deadline:
        return None, -np.inf
    try:
        def obj(flat):
            if time.time() > deadline:
                return 1e10
            return _objective_sym_full(flat)
        res = minimize(
            obj, flat_init, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if time.time() > deadline:
            return None, -np.inf
        if res.fun < 1e9:
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
    except Exception:
        pass
    return None, -np.inf

def _normalize_pts(pts):
    """Normalize points to unit convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return pts
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    return pts

def construct():
    """Place 14 points in convex position to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Initial guess: Regular 14-gon, take first 7 vertices for symmetric parameterization
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    init_params = np.column_stack([np.cos(angles), np.sin(angles)])
    init_pts = _build_symmetric_points(init_params)
    init_score = _score(init_pts)
    update_best(init_pts, init_score)
    
    # Phase 1: Differential Evolution in 14D symmetric space
    if time.time() < deadline - 60.0:
        bounds = [(-2.0, 2.0)] * (2 * M)
        try:
            def de_obj(flat):
                if time.time() > deadline - 5:
                    return 1e10
                return _objective_sym_soft(flat, beta=30.0)
            
            res_de = differential_evolution(
                de_obj, bounds,
                seed=42,
                maxiter=40,
                popsize=20,
                tol=1e-6,
                init='latinhypercube',
                polish=False,
                update_interval=1,
                callback=None
            )
            
            if time.time() < deadline - 30.0 and res_de.fun < 1e9:
                cand_pts = _build_symmetric_points(res_de.x.reshape(M, 2))
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
                
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement in symmetric space (soft-min)
    if time.time() < deadline - 40.0:
        # Refine from DE result or initial
        if best_score > init_score:
            # Extract symmetric params from best_points if it's symmetric
            # Since we built it symmetrically, we can use the first 7 points
            flat_init = best_points[:M].flatten()
        else:
            flat_init = init_params.flatten()
        
        cand_pts, cand_score = _refine_sym_lbfgs(flat_init, deadline, beta=50.0)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
    
    # Phase 3: Nelder-Mead refinement in symmetric space (hard objective)
    if time.time() < deadline - 25.0:
        flat_init = best_points[:M].flatten()
        cand_pts, cand_score = _refine_sym_nm(flat_init, deadline)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
    
    # Phase 4: Try full 28D refinement from current best
    if time.time() < deadline - 20.0:
        flat_init = best_points.flatten()
        cand_pts, cand_score = _refine_full_lbfgs(flat_init, deadline, beta=50.0)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
    
    # Phase 5: Multi-start local refinement
    rng = np.random.RandomState(123)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Perturb current best in symmetric space
        if best_points is not None:
            # Assume best is symmetric, perturb first 7
            flat_init = best_points[:M].flatten()
            noise = rng.normal(0, 0.05, (M, 2)).flatten()
            perturbed = flat_init + noise
            
            cand_pts, cand_score = _refine_sym_lbfgs(perturbed, deadline, beta=50.0)
            if cand_pts is not None and cand_score > best_score:
                update_best(cand_pts, cand_score)
    
    # Phase 6: Final Nelder-Mead polish on full space
    if time.time() < deadline - 5.0:
        flat_init = best_points.flatten()
        cand_pts, cand_score = _refine_full_nm(flat_init, deadline)
        if cand_pts is not None and cand_score > best_score:
            update_best(cand_pts, cand_score)
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_points = _normalize_pts(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
