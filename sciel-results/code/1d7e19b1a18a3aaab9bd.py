"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = 0.0
    
    # Known best is ~0.11138. We can search up to 0.115 to be safe, or higher if we want to try.
    # Let's start with a range that includes the known best.
    r_low = 0.05
    r_high = 0.115
    
    def get_score(points):
        """Calculate the actual packing radius for a set of points."""
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if not np.all(np.isfinite(pts)):
            return 0.0
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def penalty_objective_and_grad(x, r_target):
        """
        Objective: Sum of squared violations of the constraints for radius r_target.
        Constraints:
        1. Pairwise distance >= 2 * r_target
        2. Distance to boundary >= r_target (i.e., |coord| <= 0.5 - r_target)
        """
        pts = x.reshape(N, 3)
        
        # Pairwise violations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Violation amount: max(0, 2*r - d)
        # We use a smooth approximation or just direct calculation. 
        # For L-BFGS-B, direct calculation with a small epsilon in denominator is usually fine.
        # Let's use: v_ij = max(0, 2*r - d_ij)
        # Penalty = sum(v_ij^2)
        
        # To make it differentiable everywhere, we can use a soft max, 
        # but standard max(0, x) has a kink. L-BFGS-B can handle it, but gradients might be noisy.
        # A smoother penalty: 0.5 * (max(0, 2r - d)^2) is C1.
        
        # Calculate violations
        target_dist = 2.0 * r_target
        violations = np.maximum(0.0, target_dist - dists)
        np.fill_diagonal(violations, 0.0)
        
        # Objective value
        obj_val = np.sum(violations ** 2)
        
        # Gradient of pairwise part
        # d/dx_i sum_j (max(0, 2r - d_ij)^2)
        # If d_ij < 2r:
        #   d/dx_i (2r - d_ij)^2 = 2(2r - d_ij) * (-d_ij/dx_i)
        #   d_ij/dx_i = (x_i - x_j) / d_ij
        #   grad_i = -2(2r - d_ij) * (x_i - x_j) / d_ij
        
        grad = np.zeros_like(pts)
        
        # Only compute gradient where violation > 0
        mask = violations > 0
        if np.any(mask):
            # We need to be careful with d_ij = 0. 
            # If d_ij is very small, we can add a small epsilon.
            dists_safe = np.where(dists < 1e-8, 1e-8, dists)
            
            # grad_i = -2 * sum_j (violations_ij * (x_i - x_j) / d_ij)
            # Let W_ij = 2 * violations_ij / d_ij
            # grad_i = - sum_j W_ij * (x_i - x_j)
            
            W = np.zeros((N, N))
            W[mask] = 2.0 * violations[mask] / dists_safe[mask]
            
            # grad_i = - (sum_j W_ij * x_i - sum_j W_ij * x_j)
            #        = - x_i * (sum_j W_ij) + (W @ x)_i
            
            row_sums = np.sum(W, axis=1)
            grad_pair = -(row_sums[:, np.newaxis] * pts - (W @ pts))
            grad += grad_pair
        
        # Boundary violations
        # Constraint: |x_ik| <= 0.5 - r_target
        # Violation: max(0, |x_ik| - (0.5 - r_target))
        
        bound_limit = 0.5 - r_target
        abs_pts = np.abs(pts)
        b_violations = np.maximum(0.0, abs_pts - bound_limit)
        
        obj_val += np.sum(b_violations ** 2)
        
        # Gradient of boundary part
        # d/dx_ik (max(0, |x_ik| - L)^2)
        # If |x_ik| > L:
        #   2 * (|x_ik| - L) * sign(x_ik)
        
        grad_bound = np.zeros_like(pts)
        mask_b = b_violations > 0
        if np.any(mask_b):
            signs = np.sign(pts)
            # Handle sign(0) = 0, but if violation > 0, |x| > L >= 0, so x != 0 unless L<0 which is not the case.
            # So sign is well-defined and non-zero where violation > 0.
            grad_bound[mask_b] = 2.0 * b_violations[mask_b] * signs[mask_b]
            grad += grad_bound
        
        return obj_val, grad.flatten()

    def init_grid_5x5x4():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(4) + 0.5) / 4 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()

    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()

    def find_feasible_radius(r_target, x0, max_iter=2000, tol=1e-8):
        """
        Try to find a configuration for the given r_target.
        Returns (is_feasible, points, obj_val)
        """
        if time.time() > deadline - 5.0:
            return False, None, np.inf

        try:
            # Use L-BFGS-B
            # We don't strictly need bounds if we penalize boundary violations, 
            # but bounds help keep points inside the cube roughly.
            # However, if we use hard bounds, the penalty for boundary is redundant for points on the bound.
            # Let's use bounds to help the optimizer.
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            res = minimize(penalty_objective_and_grad, x0, args=(r_target,), method='L-BFGS-B', 
                           bounds=bounds, jac=True,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            final_obj = res.fun
            pts = res.x.reshape(N, 3)
            
            # Check if the solution is actually feasible
            # We check the true score to be sure
            score = get_score(pts)
            
            # If the score is close to r_target, it's feasible.
            # Due to numerical errors, it might be slightly less.
            if score >= r_target * (1 - 1e-5):
                return True, pts, final_obj
            else:
                return False, pts, final_obj
                
        except Exception:
            return False, None, np.inf

    # 1. Generate initial points
    x0_grid = init_grid_5x5x4()
    x0_rand = init_random()
    
    # 2. Coarse Binary Search to find a feasible radius
    # Start with a known good radius to ensure we have a baseline
    # Known best: 0.1113823475125
    # Let's try to verify if 0.111 is feasible first.
    
    r_test = 0.111
    feasible, pts_test, _ = find_feasible_radius(r_test, x0_grid, max_iter=1000)
    if feasible:
        best_radius = r_test
        best_points = pts_test
        # Update bounds
        r_low = r_test
        # We know 0.11138 is best known, so let's search above that.
        r_high = 0.115 
    else:
        # If even 0.111 is not found from this start, try random
        feasible, pts_test, _ = find_feasible_radius(r_test, x0_rand, max_iter=1000)
        if feasible:
            best_radius = r_test
            best_points = pts_test
            r_low = r_test
            r_high = 0.115
        else:
            # Fallback: just use the grid and its score
            pts_grid = x0_grid.reshape(N, 3)
            best_points = pts_grid
            best_radius = get_score(pts_grid)
            r_low = best_radius
            r_high = best_radius * 1.1

    # 3. Fine Binary Search
    # Number of steps depends on time remaining
    steps = 10
    for i in range(steps):
        if time.time() > deadline - 10.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        # Use the best found points as initial guess for the new radius
        # This is a "warm start"
        x0 = best_points.flatten()
        
        feasible, pts_new, _ = find_feasible_radius(r_mid, x0, max_iter=1500)
        
        if feasible:
            best_radius = r_mid
            best_points = pts_new
            r_low = r_mid
        else:
            r_high = r_mid
            
    # 4. Final Polish
    # Try to improve the best radius found with a very tight optimization
    if best_points is not None and time.time() < deadline - 5.0:
        # Run one more optimization step on the best radius found
        # to ensure it's as tight as possible
        x0 = best_points.flatten()
        try:
            bounds = [(-0.5, 0.5)] * (N * 3)
            res = minimize(penalty_objective_and_grad, x0, args=(best_radius,), method='L-BFGS-B', 
                           bounds=bounds, jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            pts_final = res.x.reshape(N, 3)
            score_final = get_score(pts_final)
            if score_final > best_radius:
                best_points = pts_final
                best_radius = score_final
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
