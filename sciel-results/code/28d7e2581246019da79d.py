"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
N_HALF = 7

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

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

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

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

def _points_from_sym_params(flat):
    """
    Reconstruct 14 centrally-symmetric points from 7 (x,y) pairs.
    The first 7 points are the given coordinates; the last 7 are their negatives.
    """
    pts_half = flat.reshape(N_HALF, 2)
    pts = np.vstack([pts_half, -pts_half])
    return pts

def _objective(flat):
    """Hard objective: minimize negative score."""
    pts = _points_from_sym_params(flat)
    return -_score(pts)

def _soft_min(areas, beta):
    """Numerically stable soft-min of areas."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        return -max_z / beta
    log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _objective_soft(flat, beta=50.0):
    """
    Smooth surrogate: minimize hull_area / soft_min(triangle_areas).
    This is differentiable and helps navigate the nonsmooth min.
    """
    pts = _points_from_sym_params(flat)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    sm = _soft_min(areas, beta)
    if sm < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / sm

def _generate_initial_configs(n_starts=20, seed=42):
    """
    Generate diverse initial configurations in the 14D symmetric space.
    Mixes regular polygon, jittered regular polygon, and random points.
    """
    rng = np.random.RandomState(seed)
    configs = []
    
    # 1. Regular 14-gon (first half)
    angles = np.linspace(0, 2 * np.pi, N_HALF, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts.flatten())
    
    # 2-10. Jittered regular polygons
    for i in range(9):
        jitter = rng.randn(N_HALF, 2) * (0.1 + 0.05 * i)
        configs.append((pts + jitter).flatten())
    
    # 11-20. Random points in unit disk, made centrally symmetric
    for i in range(10):
        r = np.sqrt(rng.rand(N_HALF))
        theta = 2 * np.pi * rng.rand(N_HALF)
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        configs.append(pts_rand.flatten())
    
    return configs

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    Uses multi-start local refinement in the 14D centrally-symmetric space.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def time_remaining():
        return deadline - time.time()

    def refine_single(flat_init, max_time=8.0):
        """
        Two-stage refinement from a single starting point:
        1. L-BFGS-B on soft objective (navigates nonsmooth landscape)
        2. Nelder-Mead on hard objective (polishes to exact min)
        """
        if time_remaining() < max_time:
            return None, -np.inf
        
        # Stage 1: Soft-min with L-BFGS-B
        try:
            res1 = minimize(
                lambda x: _objective_soft(x, beta=50.0),
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if time_remaining() < 3.0:
                return None, -np.inf
            flat_after_soft = res1.x
        except Exception:
            flat_after_soft = flat_init
        
        # Stage 2: Hard objective with Nelder-Mead for precise polishing
        try:
            res2 = minimize(
                _objective,
                flat_after_soft,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            cand_pts = _points_from_sym_params(res2.x)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            # Fallback: use soft-min result
            try:
                cand_pts = _points_from_sym_params(flat_after_soft)
                cand_score = _score(cand_pts)
                return cand_pts, cand_score
            except Exception:
                return None, -np.inf

    # --- Multi-start local refinement ---
    initial_configs = _generate_initial_configs(n_starts=20, seed=42)
    
    for i, flat_init in enumerate(initial_configs):
        if time_remaining() < 8.0:
            break
        
        cand_pts, cand_score = refine_single(flat_init, max_time=8.0)
        if cand_pts is not None and cand_score > 0:
            update_best(cand_pts, cand_score)
        
        # If we have a good result, try a few more starts with perturbation
        # around the best found so far
        if i > 0 and i % 5 == 0 and best_points is not None and time_remaining() > 20.0:
            rng = np.random.RandomState(i * 1000 + 7)
            for _ in range(2):
                if time_remaining() < 8.0:
                    break
                perturbation = rng.randn(N_HALF * 2) * 0.05
                flat_perturbed = best_points[:N_HALF].flatten() + perturbation
                cand_pts, cand_score = refine_single(flat_perturbed, max_time=8.0)
                if cand_pts is not None and cand_score > 0:
                    update_best(cand_pts, cand_score)

    # --- Final polish on the best result ---
    if best_points is not None and time_remaining() > 5.0:
        flat_best = best_points[:N_HALF].flatten()
        # One more Nelder-Mead pass with tighter tolerance
        try:
            res_final = minimize(
                _objective,
                flat_best,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            cand_pts = _points_from_sym_params(res_final.x)
            cand_score = _score(cand_pts)
            update_best(cand_pts, cand_score)
        except Exception:
            pass

    # --- Final Normalization ---
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        final_points = best_points

    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
# EVOLVE-BLOCK-END
