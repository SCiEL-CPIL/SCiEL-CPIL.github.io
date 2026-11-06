"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Stage 1: Initialization ---
    # Golden angle spiral (sunflower model) provides a good non-lattice start.
    i = np.arange(N)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    # Scale to fit inside unit circle with some margin
    r_max = 1.0 - 0.05 
    r_spiral = np.sqrt(i / N) * r_max
    theta = i * phi
    
    points = np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)])
    
    # We will optimize for a target radius r_target.
    # Start with a radius slightly below the theoretical max to ensure initial feasibility
    # Area bound: r <= 1/sqrt(N) ~ 0.0577
    # Best known: ~0.0531
    # Let's start at 0.052 and try to push higher.
    
    best_points = points.copy()
    best_score = 0.0
    
    def calc_score(pts):
        """Calculate the valid packing radius for a given set of points."""
        # Pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distance
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(norms)
        
        # The radius is the minimum of these two
        r = min(min_pair, min_bound)
        
        # Check validity: points must be strictly inside (or on boundary? 
        # Task says "on or outside ... invalid". So we need |p| < 1.
        # However, the score is defined as min(1-|c|). If |c| >= 1, score is <= 0.
        # The evaluator likely checks validity separately or treats score<=0 as invalid.
        # To be safe, we ensure our returned points are strictly inside.
        if np.max(norms) >= 1.0:
            return 0.0
            
        return r

    # Initial score
    current_score = calc_score(best_points)
    best_score = current_score

    # --- Stage 2: Barrier Method Optimization ---
    # Objective: Maximize r such that constraints are satisfied.
    # Constraints:
    # 1. d_ij >= 2r  for all i<j
    # 2. |p_i| + r <= 1 for all i
    
    # We can formulate this as maximizing the "margin" of the constraints.
    # Let f(pts) = min( min_{i<j} (d_ij/2 - r), min_i (1 - |p_i| - r) )
    # We want to maximize f(pts) + r ? No, we want to maximize r subject to f(pts) >= 0.
    # Alternatively, maximize the minimum clearance:
    # C(pts) = min( min_{i<j} d_ij/2, min_i (1 - |p_i|) )
    # This C(pts) IS the maximum possible radius for that configuration.
    # So we just need to maximize C(pts).
    
    # Since C(pts) is non-smooth (min function), we use a smooth approximation:
    # The log-sum-exp trick for the minimum:
    # min(x_1, ..., x_k) ≈ -1/beta * log(sum(exp(-b_i x_i)))
    # Here, we have two types of terms: pair distances and boundary distances.
    # Let x_1..x_M be all the "clearances" (d_ij/2 and 1-|p_i|).
    # We want to maximize min(x).
    # Equivalent to minimizing -min(x).
    # -min(x) ≈ 1/beta * log(sum(exp(-b_i x_i)))
    
    # Let's define the energy to MINIMIZE:
    # E(pts) = (1/beta) * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * (1 - |p_i|)) )
    # As beta -> infinity, E -> -min(clearances). Minimizing E maximizes the minimum clearance.
    
    # We will use a gradient-based optimization (L-BFGS-B or similar) on this smooth objective.
    # But we need gradients.
    
    # Gradient of E w.r.t p_k:
    # Let S = sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * (1 - |p_i|))
    # E = (1/beta) log S
    # dE/dp_k = (1/S) * dS/dp_k
    
    # dS/dp_k = sum_{j != k} (-beta/2) * exp(-beta * d_kj/2) * (p_k - p_j) / d_kj 
    #           + (-beta) * exp(-beta * (1 - |p_k|)) * (p_k / |p_k|)
    
    # Note: d_kj = |p_k - p_j|
    # Let u_kj = (p_k - p_j) / d_kj (unit vector from j to k)
    # dS/dp_k = sum_{j != k} (-beta/2) * w_kj * u_kj - beta * w_k * (p_k / |p_k|)
    # where w_kj = exp(-beta * d_kj/2) and w_k = exp(-beta * (1 - |p_k|))
    
    # We can compute this efficiently.
    
    def objective_and_grad(pts):
        pts = pts.reshape(N, 2)
        
        # Compute pairwise distances
        # Using cdist for speed, but we need vectors for gradient.
        # For N=300, O(N^2) is 90k ops. Very fast.
        
        # Pairwise vectors
        # diff[i, j] = p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, 1.0) # Temporary, will mask later
        
        # Weights for pairs
        # beta is a parameter. We start with moderate beta and increase it.
        # High beta makes the approximation sharper (closer to true min).
        # Low beta is smoother but less accurate.
        
        # Let's use a fixed beta that is high enough to capture the min well.
        # If r ~ 0.05, clearances are ~0.05.
        # beta ~ 100-500 should be good.
        beta = 200.0
        
        # Pair weights
        # w_ij = exp(-beta * d_ij / 2)
        # We only care about i != j.
        pair_args = -beta * dists / 2.0
        # Mask diagonal
        pair_args[np.eye(N, dtype=bool)] = -np.inf
        w_pairs = np.exp(pair_args)
        
        # Boundary weights
        norms = np.linalg.norm(pts, axis=1)
        # Avoid division by zero if point is at origin
        norms_safe = np.where(norms > 1e-8, norms, 1.0)
        bound_args = -beta * (1.0 - norms)
        w_bound = np.exp(bound_args)
        
        # Sum S
        S = np.sum(w_pairs) + np.sum(w_bound)
        
        if S < 1e-100:
            # Fallback if underflow (shouldn't happen with reasonable beta/pts)
            return 1e10, np.zeros_like(pts)
            
        E = (1.0 / beta) * np.log(S)
        
        # Gradient
        # dE/dp_k = (1/S) * dS/dp_k
        
        # dS/dp_k from pairs:
        # sum_{j != k} (-beta/2) * w_kj * (p_k - p_j) / d_kj
        # Let's compute the force vector for each point.
        
        # Unit vectors u_kj = diff[k, j] / dists[k, j]
        # diff is (N, N, 2)
        # dists is (N, N)
        # We need to divide diff by dists[..., np.newaxis]
        
        # Mask diagonal to avoid 0/0
        mask = ~np.eye(N, dtype=bool)
        
        # Safe division
        dists_safe = np.where(mask, dists, 1.0)
        unit_vectors = diff / dists_safe[..., np.newaxis]
        
        # Forces from pairs:
        # F_pair_k = sum_{j != k} (-beta/2) * w_kj * u_kj
        # Note: w_pairs is symmetric, u_kj = -u_jk.
        # We can sum over j for each k.
        
        # Multiply weights by unit vectors
        # w_pairs is (N, N)
        # unit_vectors is (N, N, 2)
        # We want sum over axis 1 (j)
        
        # Term: w_kj * u_kj
        weighted_unit = w_pairs[..., np.newaxis] * unit_vectors
        
        # Sum over j (axis 1)
        force_pairs = np.sum(weighted_unit, axis=1) # Shape (N, 2)
        
        # Apply factor (-beta/2)
        force_pairs *= (-beta / 2.0)
        
        # Forces from boundary:
        # F_bound_k = -beta * w_k * (p_k / |p_k|)
        # p_k / |p_k| is the unit radial vector.
        radial_unit = pts / norms_safe[:, np.newaxis]
        # If norm is 0, radial_unit is 0 (since pts is 0). Correct.
        force_bound = (-beta) * w_bound[:, np.newaxis] * radial_unit
        
        # Total dS/dp
        dS_dp = force_pairs + force_bound
        
        # dE/dp = (1/S) * dS_dp
        grad = dS_dp / S
        
        return E, grad

    # Optimization Loop
    # We will run several rounds of optimization with increasing beta.
    # High beta makes the landscape "spikier" around the min, which can make gradient methods
    # struggle if they hit the non-smooth transition. But starting from a good point,
    # it should converge to the local min of the min-clearance.
    
    # Since we are maximizing the minimum clearance, and the objective E is a smooth
    # approximation of -min_clearance, minimizing E is correct.
    
    # We use a simple gradient descent or L-BFGS.
    # L-BFGS is better for this scale.
    from scipy.optimize import minimize
    
    current_points = best_points.copy()
    
    # Try to improve the score by optimizing the barrier objective.
    # We will do this for a few iterations, increasing beta.
    
    betas = [50, 100, 200, 500, 1000]
    
    for beta_val in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Redefine objective with current beta
        def obj_grad(pts):
            # We need to pass beta into the function defined above.
            # Let's refactor to use a closure or global.
            # For speed, let's inline the logic or use a class.
            # Actually, let's just modify the outer function to accept beta.
            pass
        
        # To avoid re-defining functions repeatedly, let's write a helper that takes beta.
        
        def make_obj_grad(beta):
            def _obj_grad(pts):
                pts = pts.reshape(N, 2)
                
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, 1.0)
                
                pair_args = -beta * dists / 2.0
                pair_args[np.eye(N, dtype=bool)] = -np.inf
                w_pairs = np.exp(pair_args)
                
                norms = np.linalg.norm(pts, axis=1)
                norms_safe = np.where(norms > 1e-8, norms, 1.0)
                bound_args = -beta * (1.0 - norms)
                w_bound = np.exp(bound_args)
                
                S = np.sum(w_pairs) + np.sum(w_bound)
                
                if S < 1e-100:
                    return 1e10, np.zeros_like(pts)
                    
                E = (1.0 / beta) * np.log(S)
                
                mask = ~np.eye(N, dtype=bool)
                dists_safe = np.where(mask, dists, 1.0)
                unit_vectors = diff / dists_safe[..., np.newaxis]
                
                weighted_unit = w_pairs[..., np.newaxis] * unit_vectors
                force_pairs = np.sum(weighted_unit, axis=1)
                force_pairs *= (-beta / 2.0)
                
                radial_unit = pts / norms_safe[:, np.newaxis]
                force_bound = (-beta) * w_bound[:, np.newaxis] * radial_unit
                
                dS_dp = force_pairs + force_bound
                grad = dS_dp / S
                
                return E, grad
            return _obj_grad

        obj_grad_fn = make_obj_grad(beta_val)
        
        # Run L-BFGS-B
        # Bounds: x in [-1, 1], y in [-1, 1]
        bounds = [(-1.0, 1.0), (-1.0, 1.0)] * N
        
        try:
            res = minimize(
                obj_grad_fn, 
                current_points.reshape(-1), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_points = res.x.reshape(N, 2)
            
            # Check if the new points are valid and better
            new_score = calc_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    
        except Exception as e:
            # If optimization fails, keep current points
            pass

    # Final Polish:
    # The barrier method finds a local optimum of the smooth approximation.
    # We can do a few steps of direct "push apart" to refine the exact min distance.
    # This is a simple local search that only accepts moves that increase the exact min distance.
    
    for _ in range(2000):
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random point
        idx = np.random.randint(N)
        orig_pt = best_points[idx].copy()
        
        # Small random step
        step_size = 0.001
        delta = np.random.normal(0, step_size, 2)
        new_pt = orig_pt + delta
        
        # Check if inside
        if np.linalg.norm(new_pt) >= 1.0:
            continue
            
        best_points[idx] = new_pt
        
        # Calculate new score
        # To be fast, we can just check distances to neighbors?
        # For N=300, full cdist is fast enough for 2000 iters?
        # 2000 * (300^2) = 180M ops. Might be slow in Python loop.
        # Let's limit this polish to 500 iters or use a faster check.
        
        # Actually, let's just do 500 iters.
        if _ % 10 == 0:
             new_score = calc_score(best_points)
             if new_score > best_score:
                 best_score = new_score
                 current_points = best_points.copy()
             else:
                 # Revert if not better? 
                 # If we revert, we need to be careful.
                 # Let's just accept if it doesn't decrease the score significantly?
                 # No, greedy: only accept if score improves.
                 best_points[idx] = orig_pt
        else:
             # For every 10th iter we check full score. For others, we can skip check
             # or do a quick neighbor check. 
             # To keep it simple and safe, let's just check every 10th.
             # But we need to revert if the move was bad?
             # If we don't check, we might drift into a bad state.
             # Let's check every iter but optimize calc_score?
             # calc_score is O(N^2). 2000 * 90k = 180M.
             # In NumPy, 180M float ops is ~0.1-0.5s. Acceptable.
             new_score = calc_score(best_points)
             if new_score > best_score:
                 best_score = new_score
                 current_points = best_points.copy()
             else:
                 best_points[idx] = orig_pt

    # Ensure final validity
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= (1.0 - 1e-6)
    if np.any(mask):
        target_norm = 1.0 - 2e-6
        scale = target_norm / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]
        
    return best_points

# EVOLVE-BLOCK-END
