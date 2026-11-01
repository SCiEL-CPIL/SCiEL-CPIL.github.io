import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def compute_radius(pts):
        """Compute the common radius: min of half pairwise distances and boundary distances."""
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    def hex_lattice_init(n):
        """Generate hexagonal lattice points inside the unit disk."""
        # Estimate spacing
        # Area of disk = pi. Area per point ~ sqrt(3)/2 * s^2 for hex packing.
        # pi / n ~ sqrt(3)/2 * s^2 => s ~ sqrt(2*pi / (n*sqrt(3)))
        s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
        
        # Generate a grid larger than the disk
        k = int(1.0 / s) + 5
        xs = np.arange(-k, k + 1) * s
        ys = np.arange(-k, k + 1) * s * np.sqrt(3) / 2
        
        # Create meshgrid
        X, Y = np.meshgrid(xs, ys)
        
        # Shift every other row by s/2
        rows = np.arange(-k, k + 1)
        shifts = (rows % 2) * s / 2
        X = X + shifts[:, np.newaxis]
        
        pts = np.column_stack([X.ravel(), Y.ravel()])
        
        # Keep only points inside disk (with margin)
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - s / 2
        pts = pts[mask]
        
        if len(pts) < n:
            # Fallback: spiral or random if lattice too sparse (unlikely with this s)
            theta = np.linspace(0, 2 * np.pi * np.sqrt(n), n)
            r = np.sqrt(np.linspace(0, 1, n))
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        else:
            # Select n closest to center
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            pts = pts[idx]
            
        return pts

    def objective_and_grad(x_flat, r_target):
        """
        Objective: Sum of repulsion forces.
        Force model:
        1. Pairwise: If dist < 2*r_target, repel strongly.
           Use a soft plus / quadratic penalty for overlaps.
           F_pair = k * max(0, 2*r_target - dist)^2 
           Grad is w.r.t positions.
        2. Boundary: If dist_from_origin + r_target > 1, repel inward.
           F_bound = k * max(0, dist_from_origin + r_target - 1)^2
        """
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        # Compute pairwise distances
        # For N=300, 300x300 matrix is 90k floats, very fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = r_pts + r_target - 1.0
        
        # Pairwise violations
        pair_violation = 2.0 * r_target - dist
        
        # Only consider violations (where distance is too small)
        # We use a quadratic penalty for violations to make it smooth
        # Penalty = 0.5 * sum(violation^2) for violation > 0
        
        # To avoid computing gradients for non-violating pairs (optimization),
        # we can just compute on the full matrix since N is small.
        
        # Objective value
        obj = 0.5 * np.sum(np.maximum(0, pair_violation)**2)
        obj += 0.5 * np.sum(np.maximum(0, boundary_violation)**2)
        
        # Gradient
        grad = np.zeros_like(x_flat)
        
        # Pairwise Gradient
        # d/dx_i (0.5 * max(0, 2r - d_ij)^2) 
        # = max(0, 2r - d_ij) * d(2r - d_ij)/dx_i
        # = max(0, 2r - d_ij) * (-(x_i - x_j) / d_ij)
        
        # Vectorized computation
        # v_ij = max(0, 2r - d_ij)
        v = np.maximum(0, pair_violation)
        
        # Factor = v / d_ij
        # Handle d_ij = 0
        safe_dist = np.where(dist > 1e-10, dist, 1e-10)
        factor = v / safe_dist
        
        # grad_i += sum_j factor_ij * (x_i - x_j)
        # Note: d(2r - d)/dx_i = (x_i - x_j)/d. 
        # Wait, d = ||x_i - x_j||. 
        # d/dx_i (2r - d) = - (x_i - x_j) / d.
        # So d/dx_i (0.5 * v^2) = v * (- (x_i - x_j) / d) = - factor * (x_i - x_j)
        
        # Sum over j for each i
        # factor is (N, N). factor[i, j] corresponds to pair (i, j).
        # We want grad[i] = sum_j - factor[i, j] * (pts[i] - pts[j])
        # = sum_j factor[i, j] * (pts[j] - pts[i])
        
        # Let's compute sum_j factor[i, j] * pts[j] - pts[i] * sum_j factor[i, j]
        
        # sum_factor = factor.sum(axis=1)
        # weighted_sum = factor @ pts
        # grad_pairs = weighted_sum - pts * sum_factor[:, np.newaxis]
        
        sum_factor = np.sum(factor, axis=1)
        weighted_sum = factor.dot(pts)
        grad_pairs = weighted_sum - pts * sum_factor[:, np.newaxis]
        
        grad[0::2] = grad_pairs[:, 0]
        grad[1::2] = grad_pairs[:, 1]
        
        # Boundary Gradient
        # v_b = max(0, r_i + r_t - 1)
        v_b = np.maximum(0, boundary_violation)
        
        # d/dx_i (0.5 * v_b^2) = v_b * d(r_i)/dx_i = v_b * (x_i / r_i)
        
        safe_r = np.where(r_pts > 1e-10, r_pts, 1e-10)
        grad_bound = (v_b / safe_r)[:, np.newaxis] * pts
        
        grad[0::2] += grad_bound[:, 0]
        grad[1::2] += grad_bound[:, 1]
        
        return obj, grad

    # Step 1: Initialize
    pts = hex_lattice_init(N)
    best_pts = pts.copy()
    best_radius = compute_radius(pts)
    
    # Step 2: Optimize with increasing target radius
    # Start with a radius slightly smaller than the initial packing radius
    current_r = best_radius * 0.95
    
    # We will try to push current_r up towards the theoretical limit
    # Theoretical limit for 300 circles is ~ 0.05315
    # Initial hex packing is usually close to this.
    
    # Define a schedule for target radius
    # We will run L-BFGS-B for a fixed number of iterations at each target radius
    # and increase the target radius if the objective goes to 0 (or very small)
    
    max_iter_per_step = 50
    num_steps = 100 # Max steps, but we will break early
    
    x0 = best_pts.flatten()
    
    for step in range(num_steps):
        if time.time() > deadline - 10:
            break
            
        # Define objective closure with current_r
        def obj_grad(x):
            return objective_and_grad(x, current_r)
            
        # Run L-BFGS-B
        # We minimize the violation. If violation is 0, we can increase r.
        result = minimize(
            lambda x: obj_grad(x)[0],
            x0,
            jac=lambda x: obj_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter_per_step, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        x0 = result.x
        pts_new = x0.reshape(-1, 2)
        
        # Check if we achieved the target radius (i.e., no violations)
        # If result.fun is very small, it means we packed r_target.
        # We can then increase r_target.
        
        if result.fun < 1e-6:
            # Success, increase target radius
            # Increase by a small amount
            # Heuristic: if we solved it easily, increase more.
            # Let's just do a fixed small step or adaptive.
            # Current best radius is a lower bound.
            # We want to push past it.
            current_r += 0.0005 
        else:
            # Failed to reach target, maybe reduce target or just stop?
            # If we can't even maintain the current best radius, something is wrong.
            # But since we start below best_radius, we should be fine.
            # If we fail, we might be at a local minimum.
            # Let's try to increase anyway if we are close, or break if stuck.
            # Actually, if fun > 1e-6, we have overlaps at current_r.
            # We should not increase r. We should probably decrease r slightly or stop.
            # However, since we start at 0.95 * best_radius, we should be able to reach best_radius.
            # Once we reach best_radius, we start pushing higher.
            
            # If we are stuck, let's try a small perturbation to escape local minima?
            # For now, let's just break if we can't make progress.
            # But wait, result.fun > 1e-6 means we have violations.
            # This implies current_r is too large for the current configuration.
            # We should decrease current_r? No, we want to maximize r.
            # This usually means we hit a local minimum of the repulsion energy
            # that is not the global one, or the step size was too big.
            
            # Strategy: If we fail, reduce current_r slightly to get back to a valid state,
            # then try a different perturbation?
            # Simpler: Just break and return best_pts.
            # But we might have improved best_pts along the way.
            
            # Let's update best_pts if it's valid (which it should be if we started valid)
            # Actually, x0 might have violations. We should keep the last valid one.
            # Let's check radius of x0.
            r_check = compute_radius(pts_new)
            if r_check > best_radius:
                best_radius = r_check
                best_pts = pts_new.copy()
            
            # If we can't solve for current_r, we are likely stuck.
            # Let's try one more step with a slightly smaller r to see if we can recover?
            # Or just break.
            break
            
    # Final Step: Polish the best_pts found
    # The "target radius" method might leave the points in a state where they are
    # just touching or slightly overlapping due to numerical tolerance.
    # We can run one final optimization with r = best_radius to tighten them up?
    # Or just return best_pts.
    
    # Let's do a final "tightening" run with r = best_radius * 1.001 to see if we can squeeze a bit more.
    if time.time() < deadline - 5:
        # Try to push slightly beyond best_radius
        # Use best_pts as start
        x0 = best_pts.flatten()
        r_push = best_radius * 1.001
        
        def obj_grad_push(x):
            return objective_and_grad(x, r_push)
            
        result_push = minimize(
            lambda x: obj_grad_push(x)[0],
            x0,
            jac=lambda x: obj_grad_push(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if result_push.fun < 1e-8:
            pts_push = result_push.x.reshape(-1, 2)
            r_push_val = compute_radius(pts_push)
            if r_push_val > best_radius:
                best_radius = r_push_val
                best_pts = pts_push.copy()

    # Final validation
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / r_final[mask_out, np.newaxis] * 0.999999
    
    # Check for coincident points
    min_dist = np.min(pdist(best_pts))
    if min_dist < 1e-8:
        # Jitter
        np.random.seed(123)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-8:
                    best_pts[j] += np.random.normal(0, 1e-5, size=2)
                    
    return best_pts
