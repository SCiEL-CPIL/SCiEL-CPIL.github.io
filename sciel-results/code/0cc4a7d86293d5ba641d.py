import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 120s limit, leave 10s margin

    # Generate a triangular lattice
    R = 6
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    
    # Candidate pool: top 100 closest points
    pool_size = 150
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i_idx, i in enumerate(current_indices):
            if time.time() > deadline - 5:
                break
            for j in outside_candidates:
                if time.time() > deadline - 5:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
                
    best_pts = pts[current_indices]
    
    # Find closest pair in best_pts
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1: i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    # Align p0->(0,0), p1->(1,0)
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    # Also try alternative normalizations: use median distance instead of min
    # This may give a better initial configuration for optimization
    d_sorted = np.sort(d)
    med_d = d_sorted[len(d_sorted)//2]
    if med_d > 1e-8:
        transformed_med = (best_pts - p0) @ R_mat.T / med_d
        transformed_med = transformed_med[perm]
        init_params_med = transformed_med[2:].flatten()
    else:
        init_params_med = init_params
    
    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        min_d = np.min(d_opt)
        max_d = np.max(d_opt)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Smooth surrogate using log-sum-exp for soft-max and soft-min
    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max: log(sum(exp(beta * d_sq))) / beta
        max_d_sq_soft = np.log(np.sum(np.exp(beta * d_sq))) / beta
        # Soft min: -log(sum(exp(-beta * d_sq))) / beta
        min_d_sq_soft = -np.log(np.sum(np.exp(-beta * d_sq))) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    bnds = [(-10, 10)] * (2 * (N - 2))
    
    best_params = init_params
    best_score = exact_objective(init_params)
    
    # Phase 1: Smooth surrogate optimization with multiple beta values from both initializations
    initializations = [init_params, init_params_med]
    betas_to_try = [100.0, 50.0, 30.0, 20.0]
    
    for init in initializations:
        for beta in betas_to_try:
            if time.time() > deadline - 15:
                break
            def soft_obj(params, b=beta):
                pts_opt = np.zeros((N, 2))
                pts_opt[0] = [0.0, 0.0]
                pts_opt[1] = [1.0, 0.0]
                pts_opt[2:] = params.reshape(-1, 2)
                
                diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                iu = np.triu_indices(N, 1)
                d_sq = dists_sq[iu]
                
                max_d_sq_soft = np.log(np.sum(np.exp(b * d_sq))) / b
                min_d_sq_soft = -np.log(np.sum(np.exp(-b * d_sq))) / b
                
                if min_d_sq_soft < 1e-8:
                    return 1e10
                
                max_d_soft = np.sqrt(max_d_sq_soft)
                min_d_soft = np.sqrt(min_d_sq_soft)
                
                return (max_d_soft / min_d_soft)**2
            
            try:
                res = minimize(soft_obj, init, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < best_score:
                        best_params = res.x
                        best_score = exact_score
            except Exception:
                pass
    
    # Phase 2: Direct exact objective optimization with multiple restarts
    if time.time() < deadline - 10:
        # Try from current best
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
        
        # Try from slightly perturbed versions to escape local minima
        if time.time() < deadline - 8:
            rng = np.random.default_rng(42)
            for _ in range(4):
                if time.time() > deadline - 6:
                    break
                perturbed = best_params + rng.normal(0, 0.15, size=best_params.shape)
                try:
                    res = minimize(exact_objective, perturbed, method='L-BFGS-B', bounds=bnds, 
                                   options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_score:
                        best_params = res.x
                        best_score = res.fun
                except Exception:
                    pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 4:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
