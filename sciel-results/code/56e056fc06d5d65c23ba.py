import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = -1
    best_points = None
    
    # Helper to evaluate a configuration
    def get_max_multiplicity(points):
        """
        points: list of (x, y) tuples or numpy array of shape (N, 2)
        Returns: max count of any non-zero doubled area
        """
        if len(points) < 3:
            return 0
        P = np.array(points, dtype=np.int64)
        n = len(P)
        # Compute all triangle areas
        # We can do this by iterating i, j, k or using vectorization.
        # Vectorization for N=30 is easy.
        # Create indices
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        # Mask for i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        # Extract points
        Pi = P[i_flat]
        Pj = P[j_flat]
        Pk = P[k_flat]
        
        # Doubled area = | (xj - xi)(yk - yi) - (yj - yi)(xk - xi) |
        # Or determinant formula:
        # | xi(yj - yk) - xj(yi - yk) + xk(yi - yj) |
        # Let's use the cross product form which is often faster with broadcasting
        v1 = Pj - Pi
        v2 = Pk - Pi
        # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
        areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Filter out zero areas
        non_zero_areas = areas[areas > 0]
        
        if len(non_zero_areas) == 0:
            return 0
            
        # Count frequencies
        vals, counts = np.unique(non_zero_areas, return_counts=True)
        return np.max(counts)

    # Generate initial configurations
    def generate_grid(rows, cols, offset_x=0, offset_y=0, scale=1):
        points = []
        for r in range(rows):
            for c in range(cols):
                x = offset_x + c * scale
                y = offset_y + r * scale
                points.append((x, y))
        return points

    # List of initial configurations to try
    initial_configs = []
    
    # Standard 6x5 grid
    # Center it roughly in [-30, 30]
    # 6 cols, 5 rows. Width 5, Height 4.
    # Let's try different scales and offsets
    
    scales = [1, 2, 3, 4, 5, 6, 7, 8]
    rows_cols = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1), (4, 8), (8, 4)]
    
    for r, c in rows_cols:
        for s in scales:
            # Check if it fits in [-30, 30]
            width = (c - 1) * s
            height = (r - 1) * s
            if width <= 60 and height <= 60:
                # Center it
                start_x = -width // 2
                start_y = -height // 2
                # Add some random shifts to break symmetry
                for shift_x in range(-2, 3):
                    for shift_y in range(-2, 3):
                        pts = generate_grid(r, c, start_x + shift_x, start_y + shift_y, s)
                        # Ensure distinct
                        if len(set(pts)) == 30:
                            initial_configs.append(pts)
                            
    # Also add some random hexagonal-like structures
    # Hex grid logic is more complex, let's stick to distorted rectangular grids
    # We can also try adding random noise to the grid points
    
    random.seed(42)
    
    # Local Search Function
    def local_search(initial_points, max_time=5.0):
        current_points = list(initial_points)
        current_score = get_max_multiplicity(current_points)
        best_local_score = current_score
        best_local_points = current_points[:]
        
        start_ls = time.time()
        
        # Precompute all possible positions? No, just random sampling.
        # To be more efficient, we can try moving points to nearby positions first.
        
        while time.time() < max_time:
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Try a few random moves for this point
            # Bias towards small moves?
            # Let's try a random position in the grid, or a small perturbation
            if random.random() < 0.5:
                # Small perturbation
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
            else:
                # Larger jump
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                
            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
                
            # Check collision
            # This is O(N) check. We can maintain a set of occupied positions.
            # For simplicity, check list. N=30 is small.
            # But we need to be careful not to place on itself.
            if (new_x, new_y) == (old_x, old_y):
                continue
            
            # Check if (new_x, new_y) is in current_points
            # Exclude the point itself
            # Simple check:
            occupied = set(current_points)
            occupied.discard((old_x, old_y))
            if (new_x, new_y) in occupied:
                continue
                
            # Apply move
            current_points[idx] = (new_x, new_y)
            
            # Evaluate
            new_score = get_max_multiplicity(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_points = current_points[:]
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
                
        return best_local_score, best_local_points

    # Run local search on each initial config
    # Limit total time
    configs_to_try = initial_configs
    random.shuffle(configs_to_try)
    
    # We can afford to run local search for a short time on many configs
    # Or longer on fewer.
    # Let's say 2 seconds per config.
    
    for config in configs_to_try:
        if time.time() > deadline - 5.0:
            break
            
        score, points = local_search(config, max_time=2.0)
        
        if score > best_score:
            best_score = score
            best_points = points
            
    # If we have time left, do a final polish on the best point found
    if best_points is not None and time.time() < deadline - 1.0:
        score, points = local_search(best_points, max_time=deadline - time.time() - 1.0)
        if score > best_score:
            best_score = score
            best_points = points
            
    if best_points is None:
        # Fallback to a known good grid
        best_points = generate_grid(5, 6, -2, -2, 1)
        
    return best_points
