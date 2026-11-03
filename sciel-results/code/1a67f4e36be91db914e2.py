import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
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

def _normalize_points(pts):
    """Center and normalize points to max radius 1."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 1e-10:
        pts = pts / max_norm
    return pts

def _objective_general(flat_points):
    """Hard objective for general 14 points."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1.0
    pts = _normalize_points(pts)
    if np.max(np.linalg.norm(pts, axis=1)) < 1e-6:
        return 1.0
    s = _score(pts)
    return -s if s > 0 else 1.0

def _objective_soft(flat_points, beta=100.0):
    """Soft objective for general 14 points."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    pts = _normalize_points(pts)
    if np.max(np.linalg.norm(pts, axis=1)) < 1e-6:
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

def _generate_inits():
    """Generate diverse initial points for general optimization."""
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Perturbed Regular 14-gon
    for seed in range(10):
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.1, N)
        radii = np.maximum(radii, 0.5)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)
        
    # 3. Two rings: 7 outer, 7 inner
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
        pts_in = r_in * np.column_stack((np.cos(a_in), np.sin(a_in)))
        inits.append(np.vstack([pts_out, pts_in]))
        
    # 4. Three rings: 4 outer, 5 middle, 5 inner
    for r_mid, r_in in [(0.7, 0.4), (0.8, 0.5)]:
        a_out = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        a_mid = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        a_in = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 4
        pts_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
        pts_mid = r_mid * np.column_stack((np.cos(a_mid), np.sin(a_mid)))
        pts_in = r_in * np.column_stack((np.cos(a_in), np.sin(a_in)))
        inits.append(np.vstack([pts_out, pts_mid, pts_in]))
        
    # 5. Random convex-ish points (sorted by angle)
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = 0.8 + rng.uniform(0, 0.2, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)
        
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

    # === PHASE 1: Differential Evolution (Global Search) ===
    # We optimize 28 parameters.
    # Bounds: x in [-1, 1], y in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    if remaining_time() > 60.0:
        try:
            res_de = differential_evolution(
                _objective_general,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                mutation=(0.4, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            if res_de.fun < 0:
                pts14 = res_de.x.reshape(N, 2)
                pts14 = _normalize_points(pts14)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    if remaining_time() > 40.0:
        try:
            res_de2 = differential_evolution(
                _objective_general,
                bounds,
                seed=123,
                maxiter=80,
                popsize=15,
                tol=1e-5,
                mutation=(0.3, 0.8),
                recombination=0.85,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if res_de2.fun < 0:
                pts14 = res_de2.x.reshape(N, 2)
                pts14 = _normalize_points(pts14)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # === PHASE 2: Local Refinement ===
    # Use diverse inits + best from DE
    
    inits = _generate_inits()
    
    # Add best from DE if available
    if best_points is not None:
        inits.insert(0, best_points)
        
    # Add a few more random inits
    rng = np.random.default_rng(999)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = 0.7 + rng.uniform(0, 0.3, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)

    for i, init_pts in enumerate(inits):
        if remaining_time() < 15.0:
            break
        
        init_flat = init_pts.flatten()
        
        # Stage A: L-BFGS-B with soft objective
        # Start with lower beta, increase if time permits? 
        # For simplicity, use a fixed moderate beta.
        if remaining_time() > 12.0:
            try:
                res_lbfgs = minimize(
                    lambda x: _objective_soft(x, beta=100.0),
                    init_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_lbfgs.fun < 1e10:
                    pts14 = res_lbfgs.x.reshape(N, 2)
                    pts14 = _normalize_points(pts14)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
                    init_flat = res_lbfgs.x
            except Exception:
                pass

        # Stage B: Nelder-Mead on hard objective
        if remaining_time() > 8.0:
            try:
                res_nm = minimize(
                    _objective_general,
                    init_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.fun < 0:
                    pts14 = res_nm.x.reshape(N, 2)
                    pts14 = _normalize_points(pts14)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage C: Powell on hard objective
        if remaining_time() > 6.0:
            try:
                res_pow = minimize(
                    _objective_general,
                    init_flat,
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_pow.fun < 0:
                    pts14 = res_pow.x.reshape(N, 2)
                    pts14 = _normalize_points(pts14)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
