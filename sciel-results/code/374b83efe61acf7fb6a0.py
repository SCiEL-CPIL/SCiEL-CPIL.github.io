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
    # R=5 provides a good balance between candidate pool size and computation time
    R = 5 
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    n_hex = len(hex_pts)
    
    def score_points(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        if len(pts) < 2:
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        ratio_sq = max_d_sq / min_d_sq
        return ratio_sq

    # 1. Discrete Local Search on the Lattice
    # Start with the 16 closest points to the origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    # We only need to consider a subset of the lattice points for swapping
    # Points too far away will likely increase the max distance significantly
    # Let's consider the top 80 closest points as candidates
    num_candidates = min(80, n_hex)
    candidate_indices = sorted_indices[:num_candidates].tolist()
    
    initial_subset = sorted_indices[:N].tolist()
    best_score = score_points(hex_pts[initial_subset])
    best_subset = initial_subset.copy()
    
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    # Perform local search: swap one point in subset with one outside
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(N):
            if time.time() > deadline - 30:
                break
            # Get the current point in the subset at index i
            current_idx = best_subset[i]
            # Try swapping with each outside candidate
            for j in outside:
                if time.time() > deadline - 30:
                    break
                # Create new subset
                new_subset = best_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_subset = new_subset
                    in_subset = set(best_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    # Restart the outer loop to check other positions again
                    break
    
    best_pts = hex_pts[best_subset]
    
    # 2. Continuous Optimization
    # We will run the optimization on the best discrete subset and a few perturbed versions
    
    def normalize_and_fix(pts):
        """
        Translate, rotate, and scale the points such that:
        - Point 0 is at (0,0)
        - Point 1 is at (1,0)
        Returns the parameters for the remaining N-2 points.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return np.zeros((N-2)*2)

        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        return init_params

    def get_dists_from_params(params):
        """Reconstruct points from fixed points and params, return pairwise distances."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        return dists[iu]

    def smooth_objective(params, beta):
        """Smooth surrogate using log-sum-exp for max and min."""
        d = get_dists_from_params(params)
        if np.any(d < 1e-8):
            return 1e10
        max_d = np.max(d)
        min_d = np.min(d)
        
        # Numerically stable log-sum-exp
        d_shifted_max = d - max_d
        exp_terms_max = np.exp(beta * d_shifted_max)
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            return 1e10
        max_d_soft = max_d + (1.0 / beta) * np.log(sum_exp_max)
        
        d_shifted_min = d - min_d
        exp_terms_min = np.exp(-beta * d_shifted_min)
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            return 1e10
        min_d_soft = min_d - (1.0 / beta) * np.log(sum_exp_min)
        
        if min_d_soft < 1e-8:
            return 1e10
            
        ratio = max_d_soft / min_d_soft
        return ratio**2

    def exact_objective(params):
        """Exact objective for final evaluation."""
        d = get_dists_from_params(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    best_score = 1e10
    best_pts = None
    best_params = None

    bnds = [(-5, 5)] * (2 * (N - 2))

    # Generate a few initial candidates for continuous optimization
    candidates = [best_pts]
    
    # Add perturbed versions of the best discrete subset
    np.random.seed(42)
    for i in range(4):
        pts = best_pts.copy()
        noise = np.random.normal(0, 0.05, pts.shape)
        pts += noise
        candidates.append(pts)
    
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 25:
            break
            
        init_params = normalize_and_fix(init_pts)
        current_score = exact_objective(init_params)
        
        # Optimization phases
        # Phase 1: Coarse, low beta
        if time.time() < deadline - 45:
            try:
                res1 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(10.0,),
                               options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
                if res1.fun < 1e9:
                    candidate_params = res1.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 2: Medium, higher beta
        if time.time() < deadline - 25:
            try:
                res2 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(100.0,),
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                if res2.fun < 1e9:
                    candidate_params = res2.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 3: Fine, very high beta
        if time.time() < deadline - 15:
            try:
                res3 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(1000.0,),
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if res3.fun < 1e9:
                    candidate_params = res3.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Phase 4: Exact polish
        if time.time() < deadline - 5:
            try:
                res4 = minimize(exact_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res4.fun < 1e9:
                    candidate_params = res4.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()
            final_pts = np.zeros((N, 2))
            final_pts[0] = [0.0, 0.0]
            final_pts[1] = [1.0, 0.0]
            final_pts[2:] = best_params.reshape(-1, 2)
            best_pts = final_pts

    if best_pts is None:
        best_pts = hex_pts[best_subset]
        
    # Final verification
    final_score = score_points(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
