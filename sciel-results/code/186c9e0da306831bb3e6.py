import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # 1. Initialization
    # Use a 7x7 grid + 1 center point? Or 8x8 grid subset?
    # 7x7 grid spacing 1/7. Radius 1/14 ~ 0.0714.
    # Best known 0.07137.
    # Let's use a grid of 50 points.
    # Maybe a 5x10 grid?
    # Let's stick to the 7x7 grid + 1 point at (0,0) or similar, then let optimizer move them.
    
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # We have 49 points. Add one at (0,0)? (0,0) is occupied by center of 7x7 grid (index 3,3).
    # So grid already has a point at 0,0.
    # We need 50 points.
    # Let's use 8x8 grid and take 50 points?
    # Or just duplicate one? No, must be non-overlapping.
    # Let's use a 5x10 grid.
    # Spacing x: 1/5, spacing y: 1/10.
    # Radius limited by min(1/10, 1/10) = 0.1? No.
    # In 5x10, dx=0.2, dy=0.1. Min dist 0.1. Radius 0.05. Worse.
    
    # The 7x7 grid is the best regular grid for N=49.
    # For N=50, we can start with 7x7 grid and move one point or add one.
    # Actually, let's just use the 7x7 grid and add a point at (0.5, 0.5) - corner? No, boundary.
    # Let's start with 50 random points in the square, then optimize?
    # Random restarts are risky for time.
    
    # Better: Start with the 7x7 grid (49 pts) and the 50th at (0,0) is already there.
    # We need to split or move.
    # Let's use the 7x7 grid and perturb it slightly, and add a 50th point near the center, then optimize.
    # Or simply use a 8x8 grid (64 pts) and pick the 50 with largest "clearance"?
    
    # Let's try: Generate 7x7 grid. Add 50th point at (0, 0.5/7) ??
    # Let's just use 50 points from a 10x10 grid? No, too sparse.
    
    # Strategy:
    # 1. Create 7x7 grid (49 points).
    # 2. Add 50th point at (0,0) + small offset? No, (0,0) is occupied.
    # 3. Let's use 8x8 grid, take 50 points.
    # 8x8 spacing 1/8 = 0.125. Radius 0.0625.
    # This is a lower bound.
    # The optimizer will expand it.
    
    # Let's use the 7x7 grid and "split" one cell to make 50 points?
    # E.g. take the 7x7 grid, and replace one point with two points close together.
    # This gives a valid initial config with r slightly less than 1/14.
    
    # Let's do that.
    pts = []
    for i in range(7):
        for j in range(7):
            x = (i + 0.5) / 7.0 - 0.5
            y = (j + 0.5) / 7.0 - 0.5
            pts.append([x, y])
    # pts has 49 points.
    # Take the point at (0,0) which is i=3, j=3.
    # Replace it with two points: (-0.02, 0) and (0.02, 0).
    # Remove the original (0,0).
    # Find index of (0,0). It's index 3*7+3 = 24?
    # i=3, j=3 -> x=0, y=0.
    idx_center = 3 * 7 + 3
    pts.pop(idx_center)
    pts.append([-0.02, 0.0])
    pts.append([0.02, 0.0])
    
    x0 = np.array(pts)
    
    # 2. Optimization Loop
    # Define violation function
    def violation(x_flat, r):
        x = x_flat.reshape(-1, 2)
        # Boundary violation
        # dist to boundary
        d_bound = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        v_bound = np.sum(np.maximum(0, 2*r - d_bound)**2)
        
        # Pairwise violation
        # Compute all pair distances
        # 50x50 matrix
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        # We want dist >= 2r
        # Violation is max(0, 2r - dist)^2
        # Only consider i < j to avoid double counting? Sum over all is fine, just 2x larger.
        v_pair = np.sum(np.maximum(0, 2*r - dist)**2)
        
        return v_bound + v_pair
    
    def violation_grad(x_flat, r):
        # Analytical gradient for speed?
        # L-BFGS-B can do finite differences, but analytical is better.
        # For now, let's let L-BFGS-B use finite diff if we don't provide grad.
        # But with 100 vars, finite diff is 100 evaluations per step.
        # Let's provide gradient.
        
        x = x_flat.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        # Boundary
        # d_bound_i = min(0.5-|x_i|, 0.5-|y_i|)
        # If x_i > 0, d_x = 0.5 - x_i. grad_x = -1.
        # If x_i < 0, d_x = 0.5 + x_i. grad_x = 1.
        # Same for y.
        # d_bound = min(d_x, d_y)
        # If d_x < d_y, active constraint is x.
        
        d_x = 0.5 - np.abs(x[:, 0])
        d_y = 0.5 - np.abs(x[:, 1])
        d_bound = np.minimum(d_x, d_y)
        
        # Active boundary
        active_x = d_x < d_y
        active_y = d_y < d_x
        # If equal, both.
        
        # v_bound = sum (max(0, 2r - d_bound))^2
        # Let e = 2r - d_bound
        # If e > 0, term is e^2. Deriv is 2e * (-d_bound')
        # d_bound' wrt x_i:
        # if active_x:
        #   if x_i > 0: d_x = 0.5 - x_i, d/dx = -1
        #   if x_i < 0: d_x = 0.5 + x_i, d/dx = 1
        #   so d/dx = -sign(x_i)
        # if active_y:
        #   d/dy = -sign(y_i)
        
        e_bound = 2*r - d_bound
        mask_b = e_bound > 0
        if np.any(mask_b):
            # grad_x contribution
            gx = np.zeros(N)
            gy = np.zeros(N)
            # For active_x
            ax = active_x & mask_b
            if np.any(ax):
                gx[ax] = -np.sign(x[ax, 0]) * 2 * e_bound[ax]
            ay = active_y & mask_b
            if np.any(ay):
                gy[ay] = -np.sign(x[ay, 1]) * 2 * e_bound[ay]
            # If both active (tie), sum?
            # Tie case: d_x == d_y. Gradient is sum of both?
            # min(a,b) gradient is convex combination. If tie, usually 0.5/0.5 or both.
            # Let's add both for safety.
            tie = (d_x == d_y) & mask_b
            if np.any(tie):
                gx[tie] -= np.sign(x[tie, 0]) * e_bound[tie]
                gy[tie] -= np.sign(x[tie, 1]) * e_bound[tie]
            
            grad[:, 0] += gx
            grad[:, 1] += gy
        
        # Pairwise
        # v_pair = sum_{i<j} (max(0, 2r - d_ij))^2
        # d_ij = ||x_i - x_j||
        # e_ij = 2r - d_ij
        # If e_ij > 0, term e_ij^2.
        # d/dx_i e_ij = - (x_i - x_j) / d_ij
        # d/dx_i e_ij^2 = 2 e_ij * (-(x_i - x_j) / d_ij)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1.0) # Avoid div by 0
        
        e_pair = 2*r - dist
        mask_p = e_pair > 0
        # Zero out diagonal
        mask_p[np.diag_indices(N)] = False
        
        # grad for x_i is sum over j of 2 * e_ij * (-(x_i - x_j)/d_ij)
        # Note: term (i,j) and (j,i) are both in sum if we sum all.
        # My violation function summed all pairs (i,j) with i!=j.
        # So grad for x_i gets contribution from j where i!=j.
        
        # Factor: 2 * e_pair / dist
        factor = 2 * e_pair / dist
        factor[~mask_p] = 0
        
        # grad_x_i = - sum_j factor[i,j] * (x_i - x_j)
        # diff[i,j] = x_i - x_j
        # sum over j of factor[i,j] * diff[i,j]
        # This is matrix multiplication: factor @ diff
        # factor is N x N. diff is N x 2.
        # result is N x 2.
        grad_pair = -(factor @ diff)
        
        grad += grad_pair
        
        return grad.flatten()

    # Optimization
    r_current = 0.0
    # Calculate initial r
    x_curr = x0.copy()
    def calc_r(x):
        d_bound = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        r_b = np.min(d_bound)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_p = np.min(dist) / 2.0
        return min(r_b, r_p)
    
    r_current = calc_r(x_curr)
    
    # Iterative increase
    delta_r = 0.001
    max_iter = 100
    
    for k in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        r_target = r_current + delta_r
        
        # Minimize violation
        res = minimize(
            violation, 
            x_curr.flatten(), 
            args=(r_target,),
            jac=violation_grad,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        x_new = res.x.reshape(-1, 2)
        v_final = violation(x_new.flatten(), r_target)
        
        if v_final < 1e-4:
            x_curr = x_new
            r_current = r_target
            # Increase step size slightly if successful?
            # delta_r = min(0.002, delta_r * 1.05)
        else:
            # Failed to increase
            # Try smaller step?
            delta_r /= 2
            if delta_r < 1e-5:
                break
    
    # Final polish with SLSQP or tighter L-BFGS-B
    # Use SLSQP to maximize r directly?
    # Or just run L-BFGS-B again with tighter tolerance on current r?
    # Let's try to squeeze more out with SLSQP maximizing r.
    
    if time.time() < deadline - 5:
        # SLSQP setup
        def obj_slsqp(z):
            return -z[-1] # Maximize r (last var)
        
        def jac_slsqp(z):
            return np.zeros(2*N + 1)
        
        # Constraints
        # 1. Boundary
        # 2. Distance
        
        # This might be slow. Let's stick to L-BFGS-B polish.
        # Run L-BFGS-B on violation for r_current + small epsilon?
        # Or just return current best.
        
        # One more polish pass
        res = minimize(
            violation, 
            x_curr.flatten(), 
            args=(r_current + 1e-5,),
            jac=violation_grad,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_final = res.x.reshape(-1, 2)
        r_final = calc_r(x_final)
        
        if r_final > r_current:
            x_curr = x_final
            r_current = r_final

    return x_curr
