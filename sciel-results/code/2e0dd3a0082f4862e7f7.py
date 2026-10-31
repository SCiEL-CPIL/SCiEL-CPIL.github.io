import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Helper to compute exact score
    def exact_score(pts):
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    # Helper to normalize points: center at origin, scale such that sum of squared dists to origin is 1?
    # Or fix p0=(0,0) and p1=(1,0) to remove translation/rotation/scale.
    # The current code fixes p0 and p1. This reduces variables from 32 to 28.
    # Let's stick to this reduction as it is effective.
    
    def prepare_params(pts):
        # Find pair with min distance to fix? Or just fix first two?
        # Fixing arbitrary points might lead to bad local minima if they are far apart.
        # Better to find the pair with minimum distance and fix them to (0,0) and (1,0).
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_idx = np.argmin(d_sq)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            # Degenerate, return random
            return None
        
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i0 is index 0, i1 is index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # Check if transformed[0] is (0,0) and transformed[1] is (1,0)
        # Due to floating point, might be slightly off, but should be close.
        # We will optimize the remaining 14 points (28 params).
        return transformed[2:].flatten(), perm

    def params_to_pts(params, perm):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Reverse permutation
        inv_perm = np.argsort(perm)
        return pts[inv_perm]

    def soft_objective(params, beta=50.0):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Avoid zero
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
        
        return soft_max_d_sq / soft_min_d_sq

    best_score = 1e10
    best_params = None
    best_perm = None
    
    # Number of restarts depends on time
    # Each optimization takes ~10-20s. We have 145s.
    # Let's do 3-4 restarts with shorter optimization, then 1 long polish.
    
    num_restarts = 4
    time_per_restart = 30.0
    
    for r in range(num_restarts):
        if time.time() > deadline - 40: # Leave time for final polish
            break
            
        # Generate random points in a square
        pts = np.random.uniform(-1, 1, (N, 2))
        
        # Ensure distinctness
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-4:
                    pts[j] += np.random.uniform(-0.1, 0.1, 2)
        
        res_prep = prepare_params(pts)
        if res_prep is None:
            continue
        init_params, perm = res_prep
        
        # Quick optimization
        bnds = [(-5, 5)] * 28
        try:
            res = minimize(soft_objective, init_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            
            # Check time
            if time.time() > deadline - 40:
                break
                
            candidate_params = res.x
            candidate_pts = params_to_pts(candidate_params, perm)
            candidate_score = exact_score(candidate_pts)
            
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = candidate_params
                best_perm = perm
                
        except Exception as e:
            pass

    # Final Polish
    if best_params is not None and time.time() < deadline - 5:
        # Use a higher beta for sharper approximation to min/max
        beta_final = 200.0
        
        def soft_objective_final(params):
            return soft_objective(params, beta=beta_final)
        
        bnds = [(-5, 5)] * 28
        try:
            res_final = minimize(soft_objective_final, best_params, method='L-BFGS-B', bounds=bnds, 
                                 options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() < deadline - 2:
                candidate_params = res_final.x
                candidate_pts = params_to_pts(candidate_params, best_perm)
                candidate_score = exact_score(candidate_pts)
                
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_params = candidate_params
                    
        except Exception:
            pass

    # Construct final points
    if best_params is None:
        # Fallback: regular hexagonal packing
        pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append([x, y])
        pts = np.array(pts[:N])
        return pts

    final_pts = params_to_pts(best_params, best_perm)
    
    # One last check: maybe the points are not distinct?
    # The problem requires distinct points.
    # If min dist is 0, score is inf.
    # We should verify.
    
    return final_pts
