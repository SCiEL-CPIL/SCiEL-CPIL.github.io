import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal grid
    # We want 50 points. A hex grid with spacing s.
    # Let's create a larger hex grid and pick the best 50 that fit in [-0.5, 0.5]
    # Or just create a hex grid, scale it to fit, and keep 50.
    
    # Generate hexagonal lattice points
    # Rows and cols to cover the square
    # Hex spacing dx = 1, dy = sqrt(3)/2
    # To get ~50 points, we need roughly sqrt(50) ~ 7x7 grid area.
    
    def generate_hex_grid(n_points):
        # Create a grid of hex points
        # Spacing 1.0
        # We will generate a square grid of hex points and then select the center-most n_points
        # Or better: generate a hex grid that fits in [-0.5, 0.5] with optimal scaling.
        
        # Let's try a 7x7 or 8x8 grid of hex points
        # Hex grid coordinates:
        # (i * 1.0 + (j % 2) * 0.5, j * sqrt(3)/2)
        
        max_dim = 10
        points = []
        for j in range(-max_dim, max_dim + 1):
            for i in range(-max_dim, max_dim + 1):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        
        # Select the n_points closest to the origin
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:n_points]
        selected = points[indices]
        
        # Scale selected points to fit in [-0.5, 0.5]
        # Find max extent
        max_extent = np.max(np.abs(selected))
        if max_extent > 0:
            scale = 0.5 / max_extent
            selected = selected * scale * 0.98  # 0.98 to keep away from edges initially
        
        # Center the points
        center = np.mean(selected, axis=0)
        selected = selected - center
        
        # Ensure it fits in [-0.5, 0.5]
        # If scaling was too aggressive, re-scale
        max_extent = np.max(np.abs(selected))
        if max_extent > 0.5:
            selected = selected * (0.5 / max_extent)
            
        return selected

    centers = generate_hex_grid(N)
    
    # Ensure centers are strictly inside for valid initialization
    # Clip to [-0.49, 0.49] to avoid initial wall violations if scaling was off
    centers = np.clip(centers, -0.49, 0.49)
    
    # 2. Optimization: Gradient Descent on Soft-Min Objective
    
    # Objective: Minimize Sum of inverse distances^p
    # This pushes close pairs apart.
    # We also need to push circles away from walls.
    
    def compute_obj_and_grad(centers, p=20.0):
        # centers: (N, 2)
        # d_ij: distance between i and j
        # w_i: distance to nearest wall
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d_sq = np.sum(diff ** 2, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(d_sq, np.inf)
        
        # Inverse powers
        # d_sq^(-p/2)
        # To avoid division by zero, add small epsilon
        eps = 1e-12
        inv_d_p = (d_sq + eps) ** (-p / 2.0)
        
        # Sum over pairs (each pair counted twice in matrix, so divide by 2)
        obj_pairs = np.sum(inv_d_p) / 2.0
        
        # Wall distances
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # dist to left wall: x - (-0.5) = x + 0.5
        # dist to right wall: 0.5 - x
        # dist to bottom wall: y + 0.5
        # dist to top wall: 0.5 - y
        
        dx_left = centers[:, 0] + 0.5
        dx_right = 0.5 - centers[:, 0]
        dy_bottom = centers[:, 1] + 0.5
        dy_top = 0.5 - centers[:, 1]
        
        # Min distance to any wall
        # We want to maximize this. So we minimize inverse power of min distance.
        # min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
        # Using soft min again for differentiability? 
        # Or just use the actual min, but gradient is non-smooth.
        # Let's use soft min for walls too.
        
        # Soft min of wall distances
        # w_soft = (sum(w_k^(-p)))^(-1/p)
        # But we are minimizing sum(w_k^(-p)). This will push away from ALL walls, 
        # which is not ideal (we want to be close to SOME walls).
        # Actually, maximizing the minimum distance to walls means we want the 
        # "bottleneck" wall distance to be large.
        # If we minimize sum(w_k^(-p)), it will push the circle away from the 
        # closest wall, but also away from the others. This tends to center the circle.
        # For packing, we want circles in corners.
        
        # Better objective for walls: 
        # We want to maximize r_wall = min_i min_k w_{ik}.
        # This is equivalent to minimizing sum_i (min_k w_{ik})^(-p).
        # Since min_k w_{ik} is non-smooth, let's use a soft min approximation for each circle's wall distance.
        
        # For each circle i, let w_i = soft_min(dx_left_i, dx_right_i, dy_bottom_i, dy_top_i)
        # w_i = (dx_left_i^(-p) + dx_right_i^(-p) + dy_bottom_i^(-p) + dy_top_i^(-p))^(-1/p)
        # Then minimize sum_i w_i^(-p) = sum_i (dx_left_i^(-p) + ... )
        # This is exactly summing the inverse powers of the individual wall distances!
        # So the term sum_i (dx_left_i^(-p) + dx_right_i^(-p) + dy_bottom_i^(-p) + dy_top_i^(-p)) 
        # is a good objective to minimize. It penalizes being close to ANY wall.
        # Wait, if I am close to the left wall, dx_left is small, term is large. 
        # If I am close to right wall, dx_right is small, term is large.
        # This encourages being in the CENTER of the square for a single circle.
        # For multiple circles, the pairwise repulsion will push them to the edges.
        # The wall term will pull them back to the center.
        # The equilibrium should be a packing where the "pressure" from neighbors balances the wall attraction.
        
        # Let's use the sum of inverse powers of wall distances.
        w_terms = (dx_left + eps) ** (-p) + (dx_right + eps) ** (-p) + (dy_bottom + eps) ** (-p) + (dy_top + eps) ** (-p)
        obj_walls = np.sum(w_terms)
        
        total_obj = obj_pairs + obj_walls
        
        # Gradients
        # d(total_obj)/d(centers)
        
        # Gradient of obj_pairs
        # term_ij = d_ij^(-p)
        # d(term_ij)/d(c_i) = -p * d_ij^(-p-2) * (c_i - c_j)  (vector)
        # Let's compute this.
        
        # diff is (N, N, 2) where diff[i,j] = c_i - c_j
        # d_sq is (N, N)
        
        # grad_pairs[i] = sum_j -p * d_ij^(-p-2) * diff[i,j]
        # Note: d_ij^2 = d_sq[i,j]
        # d_ij^(-p-2) = (d_sq[i,j])^(-(p+2)/2)
        
        coeff = -p * (d_sq + eps) ** (-(p + 2) / 2.0)
        # Set diagonal to 0
        np.fill_diagonal(coeff, 0.0)
        
        # grad_pairs = coeff * diff
        # coeff is (N, N), diff is (N, N, 2)
        # Broadcasting: coeff[:, :, np.newaxis] * diff
        grad_pairs = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Gradient of obj_walls
        # obj_walls = sum_i [ (x_i + 0.5)^(-p) + (0.5 - x_i)^(-p) + ... ]
        
        # d/dx_i (x_i + 0.5)^(-p) = -p * (x_i + 0.5)^(-p-1)
        # d/dx_i (0.5 - x_i)^(-p) = -p * (0.5 - x_i)^(-p-1) * (-1) = p * (0.5 - x_i)^(-p-1)
        
        grad_x = -p * (dx_left + eps) ** (-p - 1) + p * (dx_right + eps) ** (-p - 1)
        grad_y = -p * (dy_bottom + eps) ** (-p - 1) + p * (dy_top + eps) ** (-p - 1)
        
        grad_walls = np.column_stack((grad_x, grad_y))
        
        total_grad = grad_pairs + grad_walls
        
        return total_obj, total_grad

    # Optimization Loop
    p = 20.0
    lr = 0.001
    best_obj, best_centers = compute_obj_and_grad(centers, p)
    
    # Adaptive learning rate
    # We will run until time limit
    
    step = 0
    while time.time() < deadline:
        obj, grad = compute_obj_and_grad(centers, p)
        
        # Check if improved
        if obj < best_obj:
            best_obj = obj
            best_centers = centers.copy()
        
        # Gradient Descent Step
        # Normalize gradient?
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Update
        centers -= lr * grad
        
        # Clip to valid range to prevent divergence
        # If a circle goes outside, the wall distance becomes negative, 
        # and the power might behave weirdly or create NaNs.
        # We should handle boundaries carefully.
        # The objective function assumes positive distances.
        # If x < -0.5, dx_left < 0. (dx_left + eps) might be negative if eps is small.
        # Negative number to a non-integer power is NaN.
        # So we MUST keep centers within the square.
        
        # Instead of clipping which creates discontinuities, 
        # we can add a strong penalty if they go out, or just clip.
        # Clipping is safer for stability.
        centers = np.clip(centers, -0.499, 0.499)
        
        # Adaptive LR
        # If objective increases, reduce LR
        # We don't track previous obj in this simple loop structure easily without state.
        # Let's just use a fixed small LR and rely on the smoothness.
        # Or implement a simple line search? Too expensive.
        
        # Let's try to increase p slowly to sharpen the minimum?
        # No, keep p fixed to avoid changing the landscape mid-opt.
        
        step += 1
        
        # Optional: Restart or random perturbation if stuck?
        # For now, simple GD.

    return best_centers

# EVOLVE-BLOCK-END
