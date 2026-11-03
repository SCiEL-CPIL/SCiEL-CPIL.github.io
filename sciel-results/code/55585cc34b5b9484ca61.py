import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

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

def _objective_soft(flat, beta=100.0):
    """Soft-min objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
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

def _objective_hard(flat):
    """Hard objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    return -_score(pts)

def _fekete_init(n, iterations=5000, seed=42):
    """
    Initialize points to maximize the product of all pairwise distances (Fekete points).
    This provides a globally well-spread starting configuration.
    """
    rng = np.random.default_rng(seed)
    # Start with a regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    # Precompute pairs for speed
    pairs = list(combinations(range(n), 2))
    
    for _ in range(iterations):
        # Pick a random point to move
        i = rng.integers(n)
        old_pt = points[i].copy()
        
        # Random perturbation
        new_pt = old_pt + rng.normal(0, 0.1, 2)
        
        # Calculate change in log-product of distances
        # log_prod = sum(log(d_ij))
        # delta = sum(log(d_new_i,j) - log(d_old_i,j)) for j != i
        
        old_dists = np.linalg.norm(points - old_pt, axis=1)
        new_dists = np.linalg.norm(points - new_pt, axis=1)
        
        # Mask self-distance
        old_dists[i] = 1.0
        new_dists[i] = 1.0
        
        # Avoid log(0)
        old_dists = np.maximum(old_dists, 1e-10)
        new_dists = np.maximum(new_dists, 1e-10)
        
        delta = np.sum(np.log(new_dists) - np.log(old_dists))
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            points[i] = new_pt
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Normalize to unit area
    hull_area = _compute_hull_area(points)
    if hull_area > 1e-10:
        points = points * np.sqrt(1.0 / hull_area)
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(pts_init):
        """Refine a set of points using SLSQP on soft objective then Nelder-Mead on hard."""
        flat_init = pts_init.flatten()
        
        # Step 1: SLSQP on soft objective
        # SLSQP handles the smooth landscape better for initialization
        try:
            res_slsqp = minimize(
                lambda x: _objective_soft(x, beta=100.0), 
                flat_init, 
                method='SLSQP', 
                options={'maxiter': 100, 'ftol': 1e-12, 'eps': 1e-6}
            )
            pts_slsqp = res_slsqp.x.reshape(N, 2)
        except Exception:
            pts_slsqp = pts_init

        # Step 2: Nelder-Mead on hard objective for precision
        try:
            res_nm = minimize(
                _objective_hard, 
                pts_slsqp.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = res_nm.x.reshape(N, 2)
            score = -res_nm.fun
            return pts_nm, score
        except Exception:
            score = _score(pts_slsqp)
            return pts_slsqp, score

    # Generate initial candidates
    candidates = []
    
    # 1. Fekete Initialization (Primary Strategy)
    # This is the key structural change: start from a globally optimal spacing configuration
    try:
        if time.time() < deadline - 30.0:
            fekete_pts = _fekete_init(N, iterations=3000, seed=1)
            candidates.append(fekete_pts)
            
            # Second Fekete run with different seed for diversity
            if time.time() < deadline - 20.0:
                fekete_pts2 = _fekete_init(N, iterations=3000, seed=2)
                candidates.append(fekete_pts2)
    except Exception:
        pass

    # 2. Regular 14-gon (Fallback/Comparison)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(regular)

    # 3. Jittered 14-gons (Diversity)
    rng = np.random.default_rng(42)
    for i in range(4):
        jitter_angles = angles + rng.normal(0, 0.05, N)
        jitter_radii = 1.0 + rng.normal(0, 0.05, N)
        jittered = np.column_stack((np.cos(jitter_angles) * jitter_radii, np.sin(jitter_angles) * jitter_radii))
        candidates.append(jittered)

    # Evaluate candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
            
        # Quick score check
        s = _score(cand)
        if s > best_score:
            update_best(cand, s)
            
        # Refine
        refined_pts, refined_score = refine(cand)
        if refined_score > best_score:
            update_best(refined_pts, refined_score)

    # Final polish of the best point found
    if best_points is not None and time.time() < deadline - 5.0:
        # Use a very tight Nelder-Mead on the best point
        try:
            res_final = minimize(
                _objective_hard, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            pts_final = res_final.x.reshape(N, 2)
            score_final = -res_final.fun
            if score_final > best_score:
                update_best(pts_final, score_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
