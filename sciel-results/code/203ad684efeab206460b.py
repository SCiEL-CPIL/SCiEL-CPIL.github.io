"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Compute pairwise distances
        n = len(centers)
        if n < 2:
            return max(0.0, min_wall)
            
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def initialize_hex_grid():
        """
        Initialize centers on a hexagonal grid.
        """
        best_centers = None
        best_r = 0.0
        
        # Try different hex grid spacings
        for d in np.linspace(0.070, 0.080, 5):
            pts = []
            row = 0
            y = -0.5 + d/2
            while y <= 0.5 - d/2:
                x_offset = 0.0 if row % 2 == 0 else d / 2.0
                x = -0.5 + d / 2.0 + x_offset
                while x <= 0.5 - d/2.0:
                    pts.append([x, y])
                    x += d
                y += d * math.sqrt(3) / 2.0
                row += 1
            
            pts = np.array(pts)
            if len(pts) < N:
                continue
            
            # Center the grid
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            pts_centered = pts - np.array([center_x, center_y])
            
            if len(pts_centered) >= N:
                # Random selection to break symmetry
                idx = np.random.choice(len(pts_centered), N, replace=False)
                init_pts = pts_centered[idx]
                
                r = compute_radius(init_pts)
                if r > best_r:
                    best_r = r
                    best_centers = init_pts.copy()
        
        if best_centers is None:
            # Fallback: Square grid
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_centers = grid[:N]
            
        return best_centers

    def multi_point_rearrangement(centers, max_time):
        """
        Strategy: Randomly select a subset of k points.
        Attempt to move them to new random positions (or slightly perturbed positions).
        Accept if the global minimum radius does not decrease.
        This allows coordinated movement of groups of circles, which single-point SA cannot do.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        t_start = time.time()
        
        # Parameters
        # We will vary the subset size k and the "aggressiveness" of the move
        # Aggressiveness can be "small perturbation" or "random restart within local region"
        
        while time.time() - t_start < max_time:
            if time.time() > deadline - 2.0:
                break
            
            # Choose subset size k
            # Smaller k (e.g., 1-5) for fine tuning, larger k (e.g., 10-20) for larger structural changes
            if np.random.random() < 0.7:
                k = np.random.randint(1, 6) # Fine tuning
            else:
                k = np.random.randint(5, 21) # Structural change
                
            if k >= N:
                k = N - 1
                
            # Select random indices
            idx = np.random.choice(N, k, replace=False)
            
            # Save original positions
            original_pos = centers[idx].copy()
            
            # Generate new positions
            # Strategy A: Small perturbation (like SA but for multiple points)
            # Strategy B: Random restart in a local box around original position
            # Strategy C: Pure random restart (risky, but might escape deep minima)
            
            move_type = np.random.random()
            if move_type < 0.5:
                # Small perturbation
                step_size = 0.01 * np.sqrt(current_r) # Scale with current radius
                new_pos = original_pos + np.random.randn(k, 2) * step_size
            elif move_type < 0.8:
                # Local box restart
                box_size = 0.05 * np.sqrt(current_r)
                new_pos = original_pos + np.random.uniform(-box_size, box_size, size=(k, 2))
            else:
                # Larger jump
                box_size = 0.1 * np.sqrt(current_r)
                new_pos = original_pos + np.random.uniform(-box_size, box_size, size=(k, 2))
            
            # Clip to square
            new_pos[:, 0] = np.clip(new_pos[:, 0], -0.5, 0.5)
            new_pos[:, 1] = np.clip(new_pos[:, 1], -0.5, 0.5)
            
            # Apply move
            centers[idx] = new_pos
            
            # Compute new radius
            new_r = compute_radius(centers)
            
            # Acceptance
            if new_r >= current_r:
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
            else:
                # Reject, revert
                centers[idx] = original_pos
                
        return best_centers, best_r

    # 1. Initialize
    centers = initialize_hex_grid()
    
    # 2. Relax using Multi-Point Rearrangement
    refined_centers, refined_r = multi_point_rearrangement(centers, max_time=135.0)
    
    # 3. Final check and return
    final_r = compute_radius(refined_centers)
    
    # Ensure validity
    if not np.all(np.isfinite(refined_centers)):
        # Fallback
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        refined_centers = grid[:N]
        
    return refined_centers

# EVOLVE-BLOCK-END
