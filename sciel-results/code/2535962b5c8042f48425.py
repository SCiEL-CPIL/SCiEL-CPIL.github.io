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

    def exact_score(pts):
        """Exact (max_d/min_d)^2 score."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d = np.sqrt(sq_dists[iu])
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-10:
            return 1e10
        return (max_d / min_d)**2

    def smooth_objective(params, beta=100.0):
        """
        Smooth surrogate using log-sum-exp.
        Objective: log(sum(exp(beta * d_i^2))) - log(sum(exp(-beta * d_i^2)))
        This approximates 2*log(max_d) - 2*log(min_d) = 2*log(max_d/min_d)
        So we return exp(2 * that) to get (max_d/min_d)^2 approximately.
        Actually, let's use: (soft_max_d / soft_min_d)^2
        soft_max_d = (1/beta) * log(sum(exp(beta * d_i)))
        soft_min_d = -(1/beta) * log(sum(exp(-beta * d_i)))
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d = np.sqrt(sq_dists[iu])
        
        # Clip to avoid numerical issues
        d = np.clip(d, 1e-10, 1e10)
        
        # Soft max: (1/beta) * log(sum(exp(beta * d)))
        beta_d = beta * d
        # For stability, subtract max
        max_beta_d = np.max(beta_d)
        soft_max_d = (1.0 / beta) * (max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d))))
        
        # Soft min: -(1/beta) * log(sum(exp(-beta * d)))
        neg_beta_d = -beta * d
        max_neg_beta_d = np.max(neg_beta_d)
        soft_min_d = -(1.0 / beta) * (max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d))))
        
        if soft_min_d < 1e-10:
            return 1e10
        
        return (soft_max_d / soft_min_d)**2

    # Generate hexagonal lattice
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    # Try multiple initial subsets and keep the best
    best_score = 1e10
    best_subset = sorted_indices[:N].tolist()
    
    # Try subsets from different "centers" by shifting
    for shift in range(0, 30):
        if time.time() > deadline - 30:
            break
        subset = sorted_indices[shift:shift+N].tolist()
        pts = hex_pts[subset]
        s = exact_score(pts)
        if s < best_score:
            best_score = s
            best_subset = subset
    
    # Local search on lattice: try swaps
    candidate_indices = sorted_indices[:150].tolist()
    in_subset = set(best_subset)
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
                new_subset = best_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = exact_score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_subset = new_subset
                    in_subset = set(best_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
    
    best_pts = hex_pts[best_subset]
    best_score = exact_score(best_pts)
    
    # Now, refine with continuous optimization using smooth surrogate
    # Fix the first two points to remove translation and rotation invariance.
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
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    # Run L-BFGS-B with smooth surrogate
    bnds = [(-10, 10)] * 28
    
    best_params = init_params.copy()
    best_smooth_score = smooth_objective(init_params, beta=50.0)
    
    # Try with different beta values
    for beta in [30.0, 50.0, 100.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(lambda p: smooth_objective(p, beta=beta), init_params, 
                          method='L-BFGS-B', bounds=bnds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_smooth_score:
                best_smooth_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Now do a final polish with the exact objective
    if time.time() < deadline - 2:
        try:
            res_exact = minimize(exact_score_wrapper, best_params,
                                method='L-BFGS-B', bounds=bnds,
                                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res_exact.fun < exact_score_from_params(best_params):
                best_params = res_exact.x.copy()
        except Exception:
            pass
    
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

def exact_score_from_params(params):
    """Compute exact score from parameters (points 0 and 1 fixed)."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(-1, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d = np.sqrt(sq_dists[iu])
    min_d = np.min(d)
    max_d = np.max(d)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d)**2

def exact_score_wrapper(params):
    """Wrapper for exact score in minimize."""
    return exact_score_from_params(params)

# EVOLVE-BLOCK-END
