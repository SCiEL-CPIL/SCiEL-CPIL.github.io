"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Hexagonal lattice initialization followed by repulsive force-based relaxation.
    
    # Step 1: Initialize on a hexagonal lattice
    # For N=200, we need to figure out a good hexagonal arrangement
    # Hexagonal packing: rows alternate offset
    # Let's estimate: sqrt(200) ~ 14.14, so roughly 14x14 or 15x14
    
    # Generate hexagonal lattice points
    # In a hexagonal lattice with spacing d, the horizontal distance between columns is d,
    # and vertical distance between rows is d * sqrt(3)/2
    # Alternate rows are offset by d/2
    
    # We'll try to fit as many points as possible in the square [-0.5, 0.5]^2
    # with some initial spacing, then relax
    
    # Start with an estimate of spacing
    # For hexagonal packing in a square of side 1:
    # If we have n_cols columns, the width is (n_cols - 1) * d
    # If we have n_rows rows, the height is (n_rows - 1) * d * sqrt(3)/2
    # We want n_cols * n_rows >= 200 (approximately, since some may be trimmed)
    
    # Let's try different configurations
    best_points = None
    best_count = 0
    
    for n_cols in range(12, 20):
        for n_rows in range(12, 20):
            if n_cols * n_rows < N:
                continue
            
            # Calculate spacing to fit in square
            # Width: (n_cols - 1) * d <= 1.0 => d <= 1.0 / (n_cols - 1)
            # Height: (n_rows - 1) * d * sqrt(3)/2 <= 1.0 => d <= 1.0 / ((n_rows - 1) * sqrt(3)/2)
            
            d_w = 1.0 / (n_cols - 1) if n_cols > 1 else 1.0
            d_h = 1.0 / ((n_rows - 1) * np.sqrt(3) / 2) if n_rows > 1 else 1.0
            d = min(d_w, d_h)
            
            if d <= 0:
                continue
            
            # Generate hexagonal lattice points
            points = []
            for row in range(n_rows):
                for col in range(n_cols):
                    x = (col - (n_cols - 1) / 2.0) * d
                    y = (row - (n_rows - 1) / 2.0) * d * np.sqrt(3) / 2
                    # Offset alternate rows
                    if row % 2 == 1:
                        x += d / 2.0
                    points.append([x, y])
            
            points = np.array(points)
            
            # Check if points are within the square (with small margin)
            margin = 0.01
            mask = (np.abs(points[:, 0]) <= 0.5 - margin) & (np.abs(points[:, 1]) <= 0.5 - margin)
            valid_points = points[mask]
            
            if len(valid_points) >= N and len(valid_points) > best_count:
                best_count = len(valid_points)
                best_points = valid_points
    
    if best_points is None or len(best_points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
    else:
        # Take first N points (or random N for diversity)
        np.random.seed(42)
        indices = np.random.choice(len(best_points), N, replace=False)
        points = best_points[indices].copy()
    
    # Step 2: Repulsive force-based relaxation
    # Use a simple iterative method: for each pair, if too close, push apart
    # Also constrain to stay within the square
    
    n_iter = 500
    learning_rate = 0.01
    
    for it in range(n_iter):
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10
        
        # Repulsive force: proportional to 1/d^2 for close pairs
        # Only consider pairs that are too close
        min_dist_target = 0.035  # Target minimum distance (2*r)
        
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < min_dist_target * 1.5:  # Only interact if somewhat close
                    # Force magnitude
                    f = learning_rate / (d ** 2 + 1e-10)
                    # Direction: push apart
                    direction = (points[i] - points[j]) / d
                    forces[i] += f * direction
                    forces[j] -= f * direction
        
        # Update positions
        points += forces
        
        # Constrain to square [-0.5, 0.5]^2
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Decay learning rate
        learning_rate *= 0.999
    
    # Final cleanup: ensure all points are strictly inside
    points[:, 0] = np.clip(points[:, 0], -0.4999, 0.4999)
    points[:, 1] = np.clip(points[:, 1], -0.4999, 0.4999)
    
    return points
# EVOLVE-BLOCK-END
