import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRI = len(IDX_I)

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
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _objective_soft(points_flat, beta=100.0):
    pts = points_flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    if np.min(_all_areas(pts)) < 1e-10:
        return 1e10
    areas = _all_areas(pts)
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-10:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_hard(points_flat):
    pts = points_flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Strategy 1: "Worst-Triangle Anchor" 
    # 1. Start with a regular 14-gon.
    # 2. Identify the 3 points forming the minimum area triangle (the "worst" triangle).
    # 3. Fix the other 11 points.
    # 4. Optimize the 3 anchor points to maximize the minimum area, 
    #    subject to the constraint that they don't create new smaller triangles 
    #    (handled by the global objective or by checking all triangles).
    #    Actually, to be safe, we optimize the 3 points to maximize the *global* minimum area, 
    #    but since we only move 3 points, we can be more targeted.
    #    However, the global minimum might change to a different triangle.
    #    Let's use a hybrid: Optimize the 3 points to maximize the area of the 
    #    specific "worst" triangle, but penalize if any other triangle becomes smaller.
    
    # Initialize with Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    current_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    update_best(current_pts, _score(current_pts))

    # Main loop: Iterate anchor optimization
    # We will try to improve the configuration by focusing on the current bottleneck.
    for iter_count in range(20):
        if time.time() > deadline - 15.0:
            break
            
        pts = best_points.copy()
        
        # Find the worst triangle
        areas = _all_areas(pts)
        min_idx = np.argmin(areas)
        anchor_i, anchor_j, anchor_k = IDX_I[min_idx], IDX_J[min_idx], IDX_K[min_idx]
        
        # Define the set of fixed points
        fixed_mask = np.ones(N, dtype=bool)
        fixed_mask[[anchor_i, anchor_j, anchor_k]] = False
        fixed_idx = np.where(fixed_mask)[0]
        
        def anchor_obj(x_anchor):
            # x_anchor is shape (3, 2)
            new_pts = pts.copy()
            new_pts[[anchor_i, anchor_j, anchor_k]] = x_anchor
            if np.any(~np.isfinite(new_pts)):
                return 1e10
            # We want to maximize the minimum area.
            # Use a soft min for smoothness, but with a high beta to be sharp.
            # Also, we should ensure the hull area is reasonable.
            return _objective_soft(new_pts.flatten(), beta=100.0)

        x0_anchor = pts[[anchor_i, anchor_j, anchor_k]]
        
        try:
            res = minimize(anchor_obj, x0_anchor.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < 1e9:
                new_pts = pts.copy()
                new_pts[[anchor_i, anchor_j, anchor_k]] = res.x.reshape(3, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    update_best(new_pts, new_score)
        except Exception:
            pass

    # Strategy 2: Full local polish with Nelder-Mead on the best found
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(_objective_hard, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            update_best(pts, score)
        except Exception:
            pass

    # Strategy 3: Multi-start random perturbations of the best structure
    if best_points is not None and time.time() < deadline - 2.0:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 2.0:
                break
            np.random.seed(seed)
            noise = np.random.normal(0, 0.05, (N, 2))
            x0 = best_points + noise
            try:
                res = minimize(_objective_soft, x0.flatten(), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
