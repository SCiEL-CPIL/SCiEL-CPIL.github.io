"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    # Generate a hexagonal lattice with enough points
    R = 4
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    
    def score_points(pts):
        # pts is an array of shape (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Local search: try swapping one point in the subset with one outside
    # We only consider points within a reasonable range
    candidate_indices = sorted_indices[:60].tolist()
    
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(N):
            if time.time() > deadline - 20:
                break
            for j in outside:
                if time.time() > deadline - 20:
                    break
                new_subset = initial_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    initial_subset = new_subset
                    in_subset = set(initial_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
    
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Refine with continuous optimization
    # Fix the first two points to remove translation and rotation invariance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair[0], pair[1]
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    # Rotation matrix
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder so that i0 is at index 0 and i1 is at index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    init_params = transformed[2:].flatten()
    
    def get_dists(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        return dists[iu]
    
    def smooth_objective(params, beta=100.0):
        """Smooth surrogate using log-sum-exp for max and min."""
        d = get_dists(params)
        if np.any(d < 1e-8):
            return 1e10
        # Soft max: (1/beta) * log(sum(exp(beta * d)))
        max_d_soft = (1.0 / beta) * np.log(np.sum(np.exp(beta * d)))
        # Soft min: -(1/beta) * log(sum(exp(-beta * d)))
        min_d_soft = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * d)))
        if min_d_soft < 1e-8:
            return 1e10
        return (max_d_soft / min_d_soft)**2
    
    def exact_objective(params):
        """Exact objective for final evaluation."""
        d = get_dists(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2
    
    bnds = [(-5, 5)] * 28
    
    best_params = init_params.copy()
    best_exact_score = exact_objective(init_params)
    
    # Phase 1: Coarse optimization with smooth surrogate
    if time.time() < deadline - 30:
        try:
            res1 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                           bounds=bnds, args=(50.0,),
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            if res1.fun < 1e9:
                candidate = res1.x
                cand_score = exact_objective(candidate)
                if cand_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = cand_score
        except Exception:
            pass
    
    # Phase 2: Medium precision with tighter smooth surrogate
    if time.time() < deadline - 20:
        try:
            res2 = minimize(smooth_objective, best_params, method='L-BFGS-B', 
                           bounds=bnds, args=(200.0,),
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < 1e9:
                candidate = res2.x
                cand_score = exact_objective(candidate)
                if cand_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = cand_score
        except Exception:
            pass
    
    # Phase 3: High-precision polish with very tight smooth surrogate
    if time.time() < deadline - 10:
        try:
            res3 = minimize(smooth_objective, best_params, method='L-BFGS-B', 
                           bounds=bnds, args=(1000.0,),
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res3.fun < 1e9:
                candidate = res3.x
                cand_score = exact_objective(candidate)
                if cand_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = cand_score
        except Exception:
            pass
    
    # Phase 4: Final exact optimization with tight tolerances
    if time.time() < deadline - 5:
        try:
            res4 = minimize(exact_objective, best_params, method='L-BFGS-B', 
                           bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res4.fun < 1e9:
                candidate = res4.x
                cand_score = exact_objective(candidate)
                if cand_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = cand_score
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify final score
    final_score = score_points(final_pts)
    if final_score < best_score:
        return final_pts
    else:
        return best_pts

# EVOLVE-BLOCK-END
