import numpy as np

N = 50

def construct():
    # 1. Initialization: Hexagonal grid
    # Hexagonal packing is denser than square.
    # We want to fit N circles.
    # Approximate side length of hex grid unit.
    # Let's just create a hex grid and trim/pad to N.
    
    def hex_grid(n):
        # Generate points in a hexagonal lattice
        # Side length of the hexagonal unit cell is roughly 1/sqrt(N)
        # Let's use a standard hex packing arrangement.
        # Rows of alternating offset.
        # Approx number of rows ~ sqrt(N)
        # Approx number of cols ~ sqrt(N)
        
        # Simple hex init:
        # dx = 1.0 / np.sqrt(N)
        # dy = dx * np.sqrt(3) / 2
        # This is for infinite plane. We need to center it.
        
        # Let's generate a large hex grid and select the central N points closest to origin.
        # Or just use a square grid rotated? No, hex is better.
        
        # Let's use a hexagonal tiling.
        # Radius of circle ~ 0.07. Diameter ~ 0.14.
        # In a hex grid with spacing s, area per circle ~ s^2 * sqrt(3)/2.
        # N * area = 1 => s ~ sqrt(2 / (N * sqrt(3)))
        
        s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
        
        # Generate grid points
        # Number of rows/cols to cover the square [-0.5, 0.5]
        # Max extent in x is 0.5.
        # Max extent in y is 0.5.
        
        # Hex grid coordinates:
        # x = s * (i + 0.5 * (j % 2))
        # y = s * j * sqrt(3) / 2
        
        # We need to generate enough points to cover the square.
        # Let's generate a square grid of hex points centered at 0.
        
        max_idx = int(np.ceil(1.0 / s)) + 2
        points = []
        for j in range(-max_idx, max_idx + 1):
            for i in range(-max_idx, max_idx + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                # Only keep points inside or close to the square
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        
        # If we have more than N points, select the N closest to the center (0,0)
        # This ensures a compact cluster.
        if len(points) > N:
            points = np.array(points)
            dists = np.sum(points**2, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        else:
            # If we have fewer, we need to add more?
            # Unlikely with max_idx calculation.
            # But to be safe, if < N, fill with random?
            # Let's assume max_idx is sufficient.
            points = np.array(points)
            if len(points) < N:
                # Add random points
                extra = np.random.uniform(-0.5, 0.5, (N - len(points), 2))
                points = np.vstack([points, extra])

        return points

    centers = hex_grid(N)
    
    # Small jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)

    # 2. Optimization: Gradient Ascent on Min Distance (Active Set Method)
    
    n_iter = 2000
    # Learning rate schedule
    lr_start = 0.02
    lr_end = 0.0001
    
    # Threshold for "active" constraints (relative to min distance)
    # We consider constraints "active" if their distance is within `delta` of the min.
    # This makes the gradient robust.
    delta_rel = 0.01
    
    for it in range(n_iter):
        t = it / (n_iter - 1)
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute wall distances
        # dist to 4 walls
        dist_x_pos = 0.5 - centers[:, 0]  # distance to right wall
        dist_x_neg = 0.5 + centers[:, 0]  # distance to left wall
        dist_y_pos = 0.5 - centers[:, 1]  # distance to top wall
        dist_y_neg = 0.5 + centers[:, 1]  # distance to bottom wall
        
        wall_dists = np.stack([dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg], axis=1) # shape (N, 4)
        min_wall_dist = np.min(wall_dists, axis=1) # shape (N,)
        
        # Global min distance
        min_pair_dist = np.min(dists)
        min_wall = np.min(min_wall_dist)
        
        r = min(min_pair_dist / 2.0, min_wall)
        
        # Identify active constraints
        # Threshold for activation
        # Use a small absolute threshold or relative to r
        # If r is small, absolute threshold should be small.
        # Let's use a fixed small absolute threshold for stability, or relative.
        # Let's use: active if dist < min_dist + 0.005
        # But min_dist changes.
        # Let's use the active set of the CURRENT minimum.
        
        # Gradient of r
        # r = min( d_pair/2, d_wall )
        # We approximate the gradient of the min function by the average gradient of all constraints that are "close" to the minimum.
        
        # Let's define "close" as: d <= min_val + epsilon
        # epsilon = 0.005 (tunable)
        eps = 0.005
        
        # Pairwise active
        # dists is (N, N). We want pairs where dists[i,j] < min_pair_dist + eps
        # And also we need to consider if the global min is a wall or a pair.
        # Actually, we should consider ALL constraints that are close to the GLOBAL minimum r.
        
        # Let's calculate the value of each constraint in terms of "radius contribution"
        # Pair (i,j) contributes d_ij / 2
        # Wall i contributes d_wall_i
        
        # Create a matrix of all constraint values
        # This is large (N^2 + 4N).
        # Instead, let's just compute gradients for pairs and walls separately and weight them.
        
        # 1. Pairwise Gradients
        # grad_i from pair (i,j) is (c_i - c_j) / (2 * d_ij)
        
        # We only want to include pairs where d_ij / 2 is close to r.
        # Condition: d_ij / 2 <= r + eps/2  => d_ij <= 2*r + eps
        # Wait, if the global min is a wall, r = min_wall. Then pairs with d_ij/2 < min_wall + eps should be active?
        # Yes, the subgradient of min(f, g) is in conv( grad f, grad g ) if f=g.
        # If f < g, it's grad f.
        # So we should include all constraints whose value is within eps of the global minimum r.
        
        # Let's define active pairs: d_ij / 2 <= r + eps
        # Active walls: d_wall_i <= r + eps
        
        # Compute weights for averaging (optional, uniform is fine)
        
        # Pairwise
        # Mask: dists <= 2 * (r + eps)
        pair_mask = dists <= 2.0 * (r + eps)
        # Ensure diagonal is false (it is inf)
        
        # Gradients for pairs
        # unit_vecs = diff / dists
        # We need to handle dists=0 (should not happen with jitter, but safe)
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        
        # Gradient contribution for pair (i,j) to center i:
        # grad_i += unit_vecs[i,j] / 2
        # But we only sum over active pairs.
        # weighted_unit_vecs = unit_vecs * pair_mask[:,:,np.newaxis]
        
        # Sum over j for each i
        grad_pairs = np.sum(unit_vecs * pair_mask[:, :, np.newaxis], axis=1) / 2.0
        
        # Wall Gradients
        # For each center i, check which wall is closest.
        # If dist_x_pos is the min for i, and dist_x_pos <= r + eps, then grad_i.x -= 1 (push left)
        # Wait, gradient of (0.5 - x) is -1.
        # We want to INCREASE the distance.
        # So we move in the direction of the gradient of the distance function.
        # grad_x of (0.5 - x) is -1.
        # So we should move in direction -1 * grad? No, gradient ascent moves in direction of gradient.
        # If we want to maximize d = 0.5 - x, we increase x? No, d decreases as x increases.
        # We want to maximize d. So we should move in direction of -grad_d?
        # Let's check:
        # d = 0.5 - x. grad_d/dx = -1.
        # Ascent: x_new = x + lr * grad_d = x - lr.
        # New d = 0.5 - (x - lr) = 0.5 - x + lr. Increased!
        # So we move in direction of gradient of distance.
        
        # For x > 0 (right wall): d = 0.5 - x, grad = -1.
        # For x < 0 (left wall): d = 0.5 + x, grad = +1.
        
        # We need to identify the active wall for each point.
        # If multiple walls are active (corner), sum their gradients?
        # Or just pick the minimum?
        # Subgradient of min is convex combo. Uniform average is fine.
        
        grad_walls = np.zeros_like(centers)
        
        # For each point, find active walls
        # wall_dists is (N, 4): [x_pos, x_neg, y_pos, y_neg]
        # Gradients:
        # x_pos (right): [-1, 0]
        # x_neg (left): [1, 0]
        # y_pos (top): [0, -1]
        # y_neg (bottom): [0, 1]
        
        # Mask for active walls
        wall_mask = wall_dists <= (r + eps) # shape (N, 4)
        
        # Sum gradients
        # grad_x component:
        # From x_pos: -1 * mask[:, 0]
        # From x_neg: 1 * mask[:, 1]
        grad_walls[:, 0] = -1.0 * wall_mask[:, 0] + 1.0 * wall_mask[:, 1]
        
        # grad_y component:
        # From y_pos: -1 * mask[:, 2]
        # From y_neg: 1 * mask[:, 3]
        grad_walls[:, 1] = -1.0 * wall_mask[:, 2] + 1.0 * wall_mask[:, 3]
        
        # Total Gradient
        # We have contributions from pairs and walls.
        # The "weight" of each constraint in the min function is roughly equal if they are all active.
        # So we can just sum the gradients?
        # Or average?
        # If we sum, the magnitude depends on the number of active constraints.
        # It's better to normalize by the number of active constraints to keep step size consistent.
        
        n_active_pairs = np.sum(pair_mask, axis=1) # per point
        n_active_walls = np.sum(wall_mask, axis=1) # per point
        
        # Total active count per point
        # Note: A point can be involved in multiple pair constraints and multiple wall constraints.
        # We just sum the gradients and divide by a normalization factor.
        # Simple approach: Just sum them. The learning rate `lr` can be tuned.
        
        total_grad = grad_pairs + grad_walls
        
        # Update
        centers += lr * total_grad
        
        # Clamp to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: Small noise to escape local minima
        if it % 100 == 0 and it > 0:
            centers += np.random.normal(0, 0.0005, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)

    return centers
