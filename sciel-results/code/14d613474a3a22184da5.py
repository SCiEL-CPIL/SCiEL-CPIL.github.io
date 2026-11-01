import numpy as np
import time

N = 200

def construct():
    """
    Constructs the centers of 200 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Strategy:
    1. Initialize points in a hexagonal grid, optimized for N=200.
    2. Use a gradient ascent approach on a smooth surrogate of the objective function.
       The objective is the minimum radius r. 
       Surrogate: S = -1/beta * log(sum(exp(-beta * d_ij))) approximates -min(d_ij).
       Maximizing S is equivalent to maximizing min(d_ij) for large beta.
       Boundary constraints are handled by adding penalty terms to the surrogate 
       or by projecting back into the square after each step.
    3. Adaptive step size: Start with a large step to escape initial grid artifacts,
       then decay the step size to refine the packing.
    4. Use multiple 'phases' with increasing beta to shift from coarse global 
       optimization to fine local refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def hex_grid_init(n):
        """
        Generate a hexagonal grid of points roughly centered in the unit square.
        """
        # Approximate number of rows
        # Area per hex cell ~ (sqrt(3)/2) * d^2. 
        # N ~ (1/d)^2 * (2/sqrt(3)) => d ~ sqrt(2/(N*sqrt(3)))
        # For N=200, d ~ sqrt(2/(200*1.732)) ~ sqrt(0.00577) ~ 0.076
        # Width ~ 1, so cols ~ 1/0.076 ~ 13.
        # Rows ~ 1 / (0.076 * sqrt(3)/2) ~ 1 / 0.066 ~ 15.
        
        rows = 14
        cols = 15
        # 14 * 15 = 210. We need 200.
        
        # Calculate spacing to fit in box
        # We want the grid to span roughly [-0.4, 0.4] initially to allow movement
        # But let's fit it to the square [-0.5, 0.5] with some margin.
        
        # Let's try to generate a grid that is dense enough.
        # dx is horizontal spacing, dy is vertical spacing.
        # dy = dx * sqrt(3) / 2
        
        # We want total width W = (cols - 1) * dx + dx/2 (for stagger) <= 1.0
        # Total height H = (rows - 1) * dy <= 1.0
        
        # Let's pick dx such that it fits.
        # If cols=15, width ~ 14.5 * dx. dx ~ 1/14.5 ~ 0.069
        # dy ~ 0.069 * 0.866 ~ 0.0597
        # Height ~ 13 * 0.0597 ~ 0.77. This leaves vertical space.
        # We can increase dx slightly to fill vertical space?
        # If we increase dx, width increases.
        
        # Let's solve for dx such that height = 1.0 (maximizing vertical usage)
        # (rows - 1) * dx * sqrt(3) / 2 = 1.0
        # dx = 2.0 / ((rows - 1) * sqrt(3))
        dx = 2.0 / ((rows - 1) * np.sqrt(3))
        dy = dx * np.sqrt(3) / 2.0
        
        # Check width:
        width = (cols - 1) * dx + dx / 2.0
        if width > 1.0:
            # Scale down
            scale = 1.0 / width
            dx *= scale
            dy *= scale
        
        points = []
        for i in range(rows):
            y = (i - (rows - 1) / 2.0) * dy
            for j in range(cols):
                x = (j - (cols - 1) / 2.0) * dx
                if i % 2 != 0:
                    x += dx / 2.0
                points.append([x, y])
        
        # We have rows * cols points. We need N.
        # Remove points from the edges to maintain symmetry?
        # Or just take the first N?
        # To be symmetric, let's remove from the corners/edges randomly but deterministically?
        # Actually, simply taking the first N might bias it.
        # Let's center the grid and then trim.
        
        pts = np.array(points)
        
        # If we have more than N, remove the ones furthest from the center?
        # No, that would leave a blob in the middle.
        # We want a rectangular-ish block.
        # Let's just take the first N in a raster scan, but center the array.
        
        if len(pts) > N:
            # Remove from the end of the list (bottom right)
            pts = pts[:N]
            
        # Center the points
        pts -= pts.mean(axis=0)
        
        # Scale to fit within [-0.5, 0.5] with a small margin
        max_extent = np.max(np.abs(pts))
        if max_extent > 0.45:
            pts *= (0.45 / max_extent)
            
        return pts

    # Initialize
    points = hex_grid_init(N)
    
    best_points = points.copy()
    best_radius = 0.0
    
    def compute_radius(pts):
        """Compute the valid radius for a set of points."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Boundary distances
        # Distance to each wall
        d_left = pts[:, 0] - (-0.5)
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] - (-0.5)
        d_top = 0.5 - pts[:, 1]
        
        min_boundary_dist = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1))
        
        # Radius is half the min of pair dist and boundary dist
        # Wait, the formula is: r = min(min_pair_dist / 2, min_boundary_dist)
        # Because circles of radius r touch if dist = 2r.
        # And circle touches wall if dist_to_wall = r.
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    # Optimization Loop
    # We use a soft-min surrogate to get gradients.
    # Objective: maximize r.
    # Surrogate: F = -1/beta * log( sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * b_i) )
    # where b_i is min distance to boundary.
    # Gradient of F w.r.t p_i gives direction to increase r.
    
    # Parameters
    # We will run several phases with increasing beta.
    # Phase 1: beta=10, steps=500, step_size=0.01
    # Phase 2: beta=50, steps=500, step_size=0.005
    # Phase 3: beta=200, steps=500, step_size=0.002
    # Phase 4: beta=1000, steps=500, step_size=0.001
    
    phases = [
        (10.0, 800, 0.015),
        (50.0, 800, 0.008),
        (200.0, 800, 0.004),
        (1000.0, 1200, 0.002),
        (5000.0, 1200, 0.001)
    ]
    
    # Precompute indices for pairs? 
    # N=200. Pairs = 200*199/2 = 19900.
    # This is small enough to compute on the fly in numpy if vectorized, 
    # but explicit indices might be faster for gradient accumulation.
    
    # Let's use vectorized operations on the full matrix for simplicity and speed in numpy.
    
    for beta, max_steps, base_step in phases:
        if time.time() > deadline:
            break
            
        step = base_step
        # Decay step size within phase? Or keep constant?
        # Let's decay slowly to allow convergence.
        
        for s in range(max_steps):
            if time.time() > deadline:
                break
                
            # Calculate distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Boundary distances
            d_left = points[:, 0] + 0.5
            d_right = 0.5 - points[:, 0]
            d_bottom = points[:, 1] + 0.5
            d_top = 0.5 - points[:, 1]
            
            # We need the gradient of the soft-min.
            # Let U = sum exp(-beta * d). 
            # F = -1/beta * log(U).
            # dF/dp_i = -1/beta * (1/U) * dU/dp_i
            # dU/dp_i = sum_{j} exp(-beta * d_ij) * d(-beta * d_ij)/dp_i
            # d(d_ij)/dp_i = -(p_i - p_j) / d_ij = (p_j - p_i) / d_ij
            # So d(-beta * d_ij)/dp_i = beta * (p_i - p_j) / d_ij
            
            # Thus, term_ij = exp(-beta * d_ij) * beta * (p_i - p_j) / d_ij
            # Note: for i=j, d=inf, exp(-inf)=0, so it vanishes.
            
            # Weights
            # To avoid overflow/underflow issues with large beta, we can subtract max?
            # But beta is max 5000. d is min ~0.07. beta*d ~ 350. exp(-350) is tiny but representable.
            # If d is very small, exp(-beta*d) is close to 1.
            
            # Calculate weights for pairs
            # We need to handle d=0 (coincident points) -> division by zero.
            # Add epsilon to denominator.
            dist_safe = np.where(dist > 1e-10, dist, 1e-10)
            
            # exp_term = exp(-beta * dist)
            # For numerical stability, we can compute this directly.
            exp_term = np.exp(-beta * dist_safe)
            np.fill_diagonal(exp_term, 0.0)
            
            # Direction (p_i - p_j) / d_ij
            # diff is p_i - p_j
            # u = diff / dist_safe
            u = diff / dist_safe[:, :, np.newaxis]
            
            # Gradient contribution from pairs:
            # grad_pair_i = sum_j exp_term[i,j] * beta * u[i,j]
            # Note: The derivative of -1/beta * log(U) w.r.t p_i is:
            # -1/beta * 1/U * sum_j exp_term[i,j] * beta * (p_i - p_j)/d_ij
            # = -1/U * sum_j exp_term[i,j] * (p_i - p_j)/d_ij
            # = -1/U * sum_j exp_term[i,j] * u[i,j]
            
            # Sum over j for each i
            sum_exp_u = np.sum(exp_term[:, :, np.newaxis] * u, axis=1)
            
            U_pairs = np.sum(exp_term)
            
            # Boundary terms
            # Objective includes boundary constraints.
            # r = min(pair_dist/2, boundary_dist).
            # This is tricky to soft-min directly with different scales.
            # Let's treat boundary distance as a 'virtual' pair with a fixed point?
            # Or just add a penalty if point is close to boundary.
            # Better: Include boundary terms in the soft-min sum.
            # Let's define effective distance to boundary b_i.
            # We want to maximize min(b_i).
            # Add terms exp(-beta * b_i) to the sum U.
            
            # b_i = min(d_left, d_right, d_bottom, d_top)
            # To soft-min the boundary distances, we can sum exp(-beta * d_wall) for each wall.
            # This effectively penalizes closeness to any wall.
            
            exp_b_left = np.exp(-beta * d_left)
            exp_b_right = np.exp(-beta * d_right)
            exp_b_bottom = np.exp(-beta * d_bottom)
            exp_b_top = np.exp(-beta * d_top)
            
            U_boundaries = np.sum(exp_b_left) + np.sum(exp_b_right) + np.sum(exp_b_bottom) + np.sum(exp_b_top)
            
            U_total = U_pairs + U_boundaries
            
            # Gradient from pairs:
            # grad_p = -1/U_total * sum_exp_u
            
            # Gradient from boundaries:
            # d/dp_i (exp(-beta * d_left)) = exp(-beta * d_left) * (-beta) * d(d_left)/dx
            # d_left = x + 0.5, d/dx = 1.
            # So term = -beta * exp_b_left.
            # Gradient of F = -1/beta * log(U) is -1/U * dU/dp.
            # dU_boundaries/dx_i = -beta * (exp_b_left - exp_b_right)  (since d_right = 0.5-x, d/dx=-1)
            # dU_boundaries/dy_i = -beta * (exp_b_bottom - exp_b_top)
            
            grad_x = -1.0 / U_total * sum_exp_u[:, 0]
            grad_y = -1.0 / U_total * sum_exp_u[:, 1]
            
            # Add boundary gradients
            # dU_bound/dx = -beta * (exp_b_left - exp_b_right)
            # dF/dx = -1/U * dU/dx = beta/U * (exp_b_left - exp_b_right)
            
            grad_x += (beta / U_total) * (exp_b_left - exp_b_right)
            grad_y += (beta / U_total) * (exp_b_bottom - exp_b_top)
            
            # Normalize gradient to unit length to use step_size directly
            grad_norm = np.linalg.norm(np.column_stack((grad_x, grad_y)), axis=1)
            # Avoid division by zero
            grad_norm_safe = np.where(grad_norm > 1e-10, grad_norm, 1.0)
            
            grad_unit_x = grad_x / grad_norm_safe
            grad_unit_y = grad_y / grad_norm_safe
            
            # Update
            new_points = points.copy()
            new_points[:, 0] += step * grad_unit_x
            new_points[:, 1] += step * grad_unit_y
            
            # Project to square
            new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
            new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
            
            # Check improvement
            # To save time, we can check every 10 steps or if step is large?
            # No, checking is fast.
            current_r = compute_radius(new_points)
            
            if current_r > best_radius:
                best_radius = current_r
                best_points = new_points.copy()
                points = new_points
            else:
                # If no improvement, we might be stuck. 
                # For now, just keep the new points to continue the trajectory?
                # Or revert? 
                # Reverting might get stuck in local minima. 
                # Let's accept if it's close? 
                # Actually, if the gradient points in a direction that increases the surrogate,
                # it should increase the radius eventually. 
                # If it decreases the actual radius, the surrogate might be misleading (beta too low).
                # But with high beta, surrogate ~ true objective.
                # Let's accept the move to maintain momentum, unless it's a huge drop?
                # Let's just update points to new_points.
                points = new_points
            
            # Decay step size slightly
            step *= 0.999
            
    return best_points

# Keep the interface
def construct():
    return construct()

# Wait, the function is defined twice? No, I just defined it once. 
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `def construct():`
# My code above defines `def construct():` inside the block? 
# No, I defined `def construct():` at the top level of the block.
# I need to make sure I don't redefine it.

# Let's re-structure to be clean.

import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def hex_grid_init(n):
        rows = 14
        cols = 15
        dx = 2.0 / ((rows - 1) * np.sqrt(3))
        dy = dx * np.sqrt(3) / 2.0
        width = (cols - 1) * dx + dx / 2.0
        if width > 1.0:
            scale = 1.0 / width
            dx *= scale
            dy *= scale
        points = []
        for i in range(rows):
            y = (i - (rows - 1) / 2.0) * dy
            for j in range(cols):
                x = (j - (cols - 1) / 2.0) * dx
                if i % 2 != 0:
                    x += dx / 2.0
                points.append([x, y])
        pts = np.array(points)
        if len(pts) > n:
            pts = pts[:n]
        pts -= pts.mean(axis=0)
        max_extent = np.max(np.abs(pts))
        if max_extent > 0.45:
            pts *= (0.45 / max_extent)
        return pts

    def compute_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        min_boundary_dist = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1))
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    points = hex_grid_init(N)
    best_points = points.copy()
    best_radius = compute_radius(points)

    phases = [
        (10.0, 800, 0.015),
        (50.0, 800, 0.008),
        (200.0, 800, 0.004),
        (1000.0, 1200, 0.002),
        (5000.0, 1200, 0.001)
    ]

    for beta, max_steps, base_step in phases:
        if time.time() > deadline:
            break
        step = base_step
        for s in range(max_steps):
            if time.time() > deadline:
                break
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            d_left = points[:, 0] + 0.5
            d_right = 0.5 - points[:, 0]
            d_bottom = points[:, 1] + 0.5
            d_top = 0.5 - points[:, 1]
            
            dist_safe = np.where(dist > 1e-10, dist, 1e-10)
            exp_term = np.exp(-beta * dist_safe)
            np.fill_diagonal(exp_term, 0.0)
            
            u = diff / dist_safe[:, :, np.newaxis]
            sum_exp_u = np.sum(exp_term[:, :, np.newaxis] * u, axis=1)
            U_pairs = np.sum(exp_term)
            
            exp_b_left = np.exp(-beta * d_left)
            exp_b_right = np.exp(-beta * d_right)
            exp_b_bottom = np.exp(-beta * d_bottom)
            exp_b_top = np.exp(-beta * d_top)
            
            U_boundaries = np.sum(exp_b_left) + np.sum(exp_b_right) + np.sum(exp_b_bottom) + np.sum(exp_b_top)
            U_total = U_pairs + U_boundaries
            
            if U_total < 1e-10:
                break
                
            grad_x = -1.0 / U_total * sum_exp_u[:, 0]
            grad_y = -1.0 / U_total * sum_exp_u[:, 1]
            
            grad_x += (beta / U_total) * (exp_b_left - exp_b_right)
            grad_y += (beta / U_total) * (exp_b_bottom - exp_b_top)
            
            grad_norm = np.linalg.norm(np.column_stack((grad_x, grad_y)), axis=1)
            grad_norm_safe = np.where(grad_norm > 1e-10, grad_norm, 1.0)
            
            grad_unit_x = grad_x / grad_norm_safe
            grad_unit_y = grad_y / grad_norm_safe
            
            new_points = points.copy()
            new_points[:, 0] += step * grad_unit_x
            new_points[:, 1] += step * grad_unit_y
            
            new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
            new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
            
            current_r = compute_radius(new_points)
            
            if current_r > best_radius:
                best_radius = current_r
                best_points = new_points.copy()
            
            points = new_points
            step *= 0.999

    return best_points
