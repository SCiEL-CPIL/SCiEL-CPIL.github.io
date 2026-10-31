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
    R = 5
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
    while improved and time.time() < deadline - 45:
        improved = False
        for i in range(N):
            if time.time() > deadline - 45:
                break
            for j in outside:
                if time.time() > deadline - 45:
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
    
    def soft_objective(params, beta=100.0):
        """Smooth surrogate using log-sum-exp for max and -log-sum-exp for min on distances."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Avoid zero distances
        d_sq = np.maximum(d_sq, 1e-12)
        d = np.sqrt(d_sq)
        
        # Soft max of distances
        max_d = np.max(d)
        soft_max_d = max_d + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d - max_d))))
        
        # Soft min of distances
        min_d = np.min(d)
        soft_min_d = min_d - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d - min_d))))
        
        if soft_min_d < 1e-8:
            return 1e10
        
        # We want to minimize (max_d / min_d)^2
        return (soft_max_d / soft_min_d)**2
    
    def epigraph_soft_objective(params, beta=100.0):
        """Epigraph formulation with soft constraints: minimize t + penalty for violations."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        d = np.sqrt(d_sq)
        
        min_d = np.min(d)
        max_d = np.max(d)
        
        if min_d < 1e-8:
            return 1e10
        
        # t = (max_d / min_d)^2
        t = (max_d / min_d)**2
        
        # Penalty for distances that are too large relative to min
        # We want all d_i <= max_d, which is automatic
        # The real constraint is that min_d should be as large as possible
        # Soft penalty: sum of exp(beta * (d_i - min_d)) for d_i > min_d
        excess = np.maximum(d - min_d, 0)
        penalty = np.sum(np.exp(beta * excess / max_d)) / len(d)
        
        return t + 0.1 * penalty
    
    def exact_objective(params):
        """Exact objective for final evaluation."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_d = np.min(d)
        max_d = np.max(d)
        
        if min_d < 1e-8:
            return 1e10
        
        return (max_d / min_d)**2
    
    bnds = [(-6, 6)] * 28
    
    best_params = init_params.copy()
    best_exact_score = exact_objective(init_params)
    
    # Phase 1: Optimize smooth surrogate with adaptive beta
    try:
        if time.time() < deadline - 35:
            # Start with lower beta for smoother landscape, then increase
            for beta_val in [20.0, 50.0, 100.0, 200.0, 400.0]:
                if time.time() > deadline - 35:
                    break
                res = minimize(soft_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10},
                               args=(beta_val,))
                
                candidate_score = exact_objective(res.x)
                if candidate_score < best_exact_score:
                    best_params = res.x
                    best_exact_score = candidate_score
                        
    except Exception:
        pass
    
    # Phase 2: Try epigraph soft objective
    try:
        if time.time() < deadline - 25:
            for beta_val in [50.0, 100.0]:
                if time.time() > deadline - 25:
                    break
                res = minimize(epigraph_soft_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8},
                               args=(beta_val,))
                
                candidate_score = exact_objective(res.x)
                if candidate_score < best_exact_score:
                    best_params = res.x
                    best_exact_score = candidate_score
                        
    except Exception:
        pass
    
    # Phase 3: Polish with exact objective from best smooth solution
    try:
        if time.time() < deadline - 15:
            res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            candidate_score = exact_objective(res2.x)
            if candidate_score < best_exact_score:
                best_params = res2.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    # Phase 4: Multiple restarts with moderate perturbations to escape local minima
    try:
        if time.time() < deadline - 10:
            rng = np.random.default_rng(42)
            n_restarts = 12
            for restart in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                    
                # Adaptive perturbation scale
                scale = 0.03 * (1 + restart * 0.25)
                perturbation = rng.normal(0, scale, size=28)
                perturbed_params = best_params + perturbation
                perturbed_params = np.clip(perturbed_params, -6, 6)
                
                # Use smooth objective for restarts, then polish
                res3 = minimize(soft_objective, perturbed_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-9},
                               args=(200.0,))
                
                candidate_score = exact_objective(res3.x)
                if candidate_score < best_exact_score:
                    best_params = res3.x
                    best_exact_score = candidate_score
                        
    except Exception:
        pass
    
    # Phase 5: Final high-precision polish
    try:
        if time.time() < deadline - 4:
            res4 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14})
            
            candidate_score = exact_objective(res4.x)
            if candidate_score < best_exact_score:
                best_params = res4.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    # Phase 6: Final ultra-precise polish if time permits
    try:
        if time.time() < deadline - 1:
            res5 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-16})
            
            candidate_score = exact_objective(res5.x)
            if candidate_score < best_exact_score:
                best_params = res5.x
                best_exact_score = candidate_score
                    
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

# EVOLVE-BLOCK-END
