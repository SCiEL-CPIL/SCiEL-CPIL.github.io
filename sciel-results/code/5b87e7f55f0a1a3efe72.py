"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(x):
    """Compute the packing radius for centers x of shape (N, 2) in [-0.5, 0.5]^2."""
    # x is (N, 2)
    # Wall clearance
    wall = 0.5 - np.max(np.abs(x), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances
    # Use a vectorized approach but it's O(N^2) = 40000 pairs, fine for N=200
    # Compute all pairwise distances
    # To save memory and time, we can do it in chunks or just compute the full matrix
    # For N=200, a 200x200 matrix is small.
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    safety_margin = 10.0
    
    # Initial hexagonal packing
    # 200 circles. A hexagonal grid with spacing 2r.
    # Let's start with a reasonable r estimate.
    # Area of square is 1. Area of circles is N * pi * r^2.
    # 200 * pi * r^2 <= 1 => r <= sqrt(1 / (200*pi)) ≈ 0.0399.
    # Best known is 0.0366.
    # Let's initialize with a hexagonal grid that fits.
    # Number of rows/cols for hexagonal packing.
    # Let's try to generate a hexagonal grid of 200 points.
    
    # We will use a local search (gradient ascent) to maximize the minimum distance.
    
    # 1. Generate initial positions
    # Hexagonal grid
    # Let's estimate the number of rows.
    # For a hexagonal grid, the density is higher.
    # Let's just use a square grid first, then optimize.
    # Actually, a hexagonal initial guess is better.
    
    # Let's create a hexagonal lattice.
    # Row 0: x = 0, 1, 2...
    # Row 1: x = 0.5, 1.5...
    # Spacing in x: 1. Row spacing in y: sqrt(3)/2.
    # We want to fit 200 points.
    
    # Let's try a square grid first for simplicity, then let the optimizer do the work.
    # The optimizer should be able to shift points into a hexagonal arrangement.
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x = grid[:N].copy()
    
    # Normalize x to be in [-0.5, 0.5]
    # The grid above is already in [-0.5, 0.5] if m is large enough.
    # Let's check the range.
    # ticks are in [-0.5, 0.5].
    
    # 2. Local search
    # We want to maximize r = min(wall_clearance, min_pair_dist/2)
    # This is equivalent to maximizing the minimum of a set of smooth functions (except for the min).
    # We can use a soft-min approximation or just do a simple coordinate ascent / gradient ascent.
    
    # Let's use a simple gradient ascent with a soft-min.
    # f(x) = -log(sum(exp(-alpha * d_i))) / alpha  approximates min(d_i)
    # where d_i are the distances (wall and pair).
    
    # However, computing all distances every iteration is O(N^2).
    # For N=200, 40000 pairs. 40000 * 200 iterations = 8 million operations. Fast.
    
    # Let's implement a simple gradient ascent.
    
    def soft_min(vals, alpha):
        # vals is a 1D array of distances
        # We want to approximate min(vals)
        # log-sum-exp trick for stability
        m = np.min(vals)
        return m - np.log(np.sum(np.exp(alpha * (vals - m)))) / alpha
    
    def gradient_soft_min(x, alpha):
        # Returns gradient of -soft_min(distances) with respect to x
        # d_i = distance_i
        # soft_min = min - log(sum(exp(alpha*(d_i - min))))/alpha
        # This is getting complicated to derive analytically.
        
        # Let's use a simpler approach:
        # Maximize the minimum distance.
        # We can use the fact that the gradient of the minimum distance is the average of the gradients of the closest distances.
        
        pass

    # Simpler approach: Random restarts + local search
    # Or just a simple coordinate ascent.
    
    # Let's try a simple gradient ascent with a finite difference approximation of the gradient of the min distance.
    # This is slow but robust.
    
    # Better: Use the subgradient of the min function.
    # The subgradient of min(d_i) is the average of the unit vectors pointing in the direction of increasing d_i for the active constraints (those with d_i = min).
    
    def get_subgradient(x, r):
        # r is the current minimum distance (wall or pair)
        # We want to move points to increase the minimum distance.
        # grad = sum_{i in active} grad(d_i) / |active|
        
        # Wall constraints
        # d_wall_i = 0.5 - max(|x_i|, |y_i|)
        # This is not smooth.
        
        # Let's just use a simple heuristic:
        # For each point, if it's close to the wall, push it in.
        # If it's close to another point, push it away.
        
        grad = np.zeros_like(x)
        
        # Wall
        for i in range(N):
            # Distance to walls
            dx = 0.5 - x[i, 0]
            dy = 0.5 - x[i, 1]
            # If close to right wall
            if dx < 2*r:
                grad[i, 0] += -1.0 / (dx + 1e-9) # Push left
            # If close to left wall
            if dx > 1 - 2*r: # x[i,0] < -0.5 + 2r
                # Actually, dx = 0.5 - x. If x is negative, dx > 0.5.
                # Let's use signed distance.
                pass
        
        # This is getting messy. Let's use a simpler, more standard approach:
        # Maximize the sum of squared distances to the nearest neighbors? No.
        
        # Let's use the "repulsion" model.
        # Force on each point is the sum of repulsive forces from all other points and walls.
        # F_i = sum_j (x_i - x_j) / ||x_i - x_j||^3  (Coulomb-like)
        # This maximizes the minimum distance in a smooth way.
        
        # Let's implement this.
        
        pass

    # Let's try a different strategy:
    # 1. Start with a hexagonal grid.
    # 2. Use a local search that tries to increase the minimum distance by moving points.
    
    # Hexagonal grid initialization
    # We want to pack 200 circles.
    # Let's estimate the grid size.
    # For a hexagonal grid, the number of points in a triangular region of side n is n(n+1)/2.
    # Or we can just generate a large hexagonal grid and take the first 200.
    
    # Let's generate a hexagonal grid with spacing s.
    # We want to fit it in [-0.5, 0.5].
    # Let's try s = 0.04 (r = 0.02).
    # Then adjust s to fit exactly 200 points.
    
    # Actually, let's just use the square grid and let the optimizer do the work.
    # The optimizer will find the hexagonal arrangement.
    
    # Let's implement a simple gradient ascent with the repulsion model.
    
    def repulsion_gradient(x):
        # Returns gradient of the sum of pairwise repulsion potentials.
        # Potential: -1/d. Gradient: (x_i - x_j) / d^3.
        # We want to maximize the minimum distance, so we want to maximize the sum of repulsions?
        # No, maximizing the sum of repulsions tends to maximize the average distance, not the minimum.
        # But it's a good heuristic.
        
        grad = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d_vec = x[i] - x[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-9: continue
                # Force is repulsive
                grad[i] += d_vec / (d**3)
        
        # Wall repulsion
        for i in range(N):
            # Right wall
            d = 0.5 - x[i, 0]
            if d < 1e-9: d = 1e-9
            grad[i, 0] += 1.0 / (d**3)
            # Left wall
            d = 0.5 + x[i, 0]
            if d < 1e-9: d = 1e-9
            grad[i, 0] -= 1.0 / (d**3)
            # Top wall
            d = 0.5 - x[i, 1]
            if d < 1e-9: d = 1e-9
            grad[i, 1] += 1.0 / (d**3)
            # Bottom wall
            d = 0.5 + x[i, 1]
            if d < 1e-9: d = 1e-9
            grad[i, 1] -= 1.0 / (d**3)
            
        return grad

    # This is O(N^2) per iteration. 40000 operations.
    # Let's vectorize it.
    
    def vectorized_repulsion_gradient(x):
        # x is (N, 2)
        # Compute all pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] # (N, N, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2)) # (N, N)
        np.fill_diagonal(dist, 1e9) # Avoid division by zero
        # Force: diff / dist^3
        # dist is (N, N), diff is (N, N, 2)
        # We need to divide diff by dist^3, but dist is 2D.
        # dist[:, :, np.newaxis] is (N, N, 1)
        force = diff / (dist[:, :, np.newaxis] ** 3)
        # Sum over j (axis 1)
        grad = np.sum(force, axis=1)
        
        # Wall repulsion
        # Right wall: d = 0.5 - x[:, 0]
        d_right = 0.5 - x[:, 0]
        d_left = 0.5 + x[:, 0]
        d_top = 0.5 - x[:, 1]
        d_bottom = 0.5 + x[:, 1]
        
        # Clip to avoid singularity
        d_right = np.clip(d_right, 1e-9, None)
        d_left = np.clip(d_left, 1e-9, None)
        d_top = np.clip(d_top, 1e-9, None)
        d_bottom = np.clip(d_bottom, 1e-9, None)
        
        grad[:, 0] += 1.0 / (d_right**3) - 1.0 / (d_left**3)
        grad[:, 1] += 1.0 / (d_top**3) - 1.0 / (d_bottom**3)
        
        return grad

    # Now, let's do gradient ascent.
    # We want to maximize the repulsion potential, which is a proxy for the minimum distance.
    # Actually, we want to maximize the minimum distance directly.
    # Let's use a combination:
    # 1. Use the repulsion gradient to move points.
    # 2. Check the minimum distance.
    # 3. If it increased, keep the move.
    
    # Let's implement a simple line search.
    
    best_x = x.copy()
    best_r = _score(best_x)
    
    # Learning rate
    lr = 0.001
    
    # Iterations
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline - safety_margin:
            break
        
        grad = vectorized_repulsion_gradient(x)
        
        # Normalize gradient?
        # The gradient magnitude can vary.
        # Let's just use a fixed learning rate.
        
        new_x = x + lr * grad
        
        # Project new_x to be inside the square
        # If a point goes outside, push it back.
        # Actually, the wall repulsion should prevent this.
        # But let's be safe.
        new_x = np.clip(new_x, -0.5, 0.5)
        
        # Check if the new position is better
        new_r = _score(new_x)
        
        if new_r > best_r:
            best_x = new_x.copy()
            best_r = new_r
            # Increase learning rate slightly?
            # lr *= 1.05
        else:
            # Decrease learning rate
            lr *= 0.9
            # If lr is too small, stop
            if lr < 1e-6:
                break
        
        x = new_x
        
    return best_x

# EVOLVE-BLOCK-END
