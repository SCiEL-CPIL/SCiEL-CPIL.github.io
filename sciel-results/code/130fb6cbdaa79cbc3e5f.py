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

    def get_bottleneck_indices(centers, r):
        """
        Identify indices of points that are limiting the radius.
        Returns a set of indices.
        """
        bottleneck = set()
        
        # Wall bottlenecks
        wall_dists = 0.5 - np.abs(centers)
        # Points where wall distance is close to r
        # If min_wall is the limiting factor, these are the bottlenecks
        min_wall = np.min(wall_dists)
        if min_wall <= r + 1e-9:
            # Find points where wall dist is close to min_wall
            # Actually, if r is determined by min_pair/2, wall might be larger.
            # We want points that *could* be moved to improve r.
            # If a point is at distance d from another, and d=2r, moving it helps.
            # If a point is at distance r from wall, moving it helps.
            
            # Let's find all points that are "tight".
            # Tight wall: wall_dist <= r + 1e-6
            tight_wall_mask = wall_dists <= (r + 1e-6)
            bottleneck.update(np.where(tight_wall_mask)[0])
            
        # Pair bottlenecks
        if N > 1:
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            # Pairs where distance is close to 2r
            tight_pair_mask = dist_matrix <= (2*r + 1e-6)
            # Indices involved in tight pairs
            tight_rows = np.where(tight_pair_mask.any(axis=1))[0]
            bottleneck.update(tight_rows)
            
        return bottleneck

    def initialize_hex_grid():
        """
        Initialize centers on a hexagonal grid.
        We generate a slightly denser grid and randomly select N points to break symmetry.
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
                while x <= 0.5 - d / 2.0:
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

    def simulated_annealing(centers, max_time):
        """
        Stochastic Local Search with Simulated Annealing.
        Focuses moves on bottleneck points to escape local minima.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        # SA Parameters
        temp = 0.01
        cooling_rate = 0.999
        min_temp = 1e-6
        
        # Move size parameters
        base_step = 0.005
        
        t_start = time.time()
        
        while time.time() - t_start < max_time and temp > min_temp:
            if time.time() > deadline - 2.0:
                break
                
            # Identify bottleneck points
            bottleneck = get_bottleneck_indices(centers, current_r)
            
            # If no bottlenecks found (shouldn't happen if r is valid), pick random
            if not bottleneck:
                bottleneck = {np.random.randint(N)}
            
            # Select a random point from the bottleneck set
            idx = list(bottleneck)[np.random.randint(len(bottleneck))]
            
            # Generate a random move for this point
            # Bias move direction: if on wall, push in. If crowded, push away from nearest neighbor.
            move = np.random.randn(2) * base_step * temp
            
            # Heuristic bias:
            # If x is close to boundary, push towards center
            if centers[idx, 0] > 0:
                move[0] -= 0.5 * base_step * temp
            elif centers[idx, 0] < 0:
                move[0] += 0.5 * base_step * temp
                
            if centers[idx, 1] > 0:
                move[1] -= 0.5 * base_step * temp
            elif centers[idx, 1] < 0:
                move[1] += 0.5 * base_step * temp
            
            new_centers = centers.copy()
            new_centers[idx] += move
            
            # Project to square
            new_centers[idx, 0] = np.clip(new_centers[idx, 0], -0.5, 0.5)
            new_centers[idx, 1] = np.clip(new_centers[idx, 1], -0.5, 0.5)
            
            new_r = compute_radius(new_centers)
            
            # Acceptance Criterion
            # Always accept if better.
            # Accept worse with probability exp(-(current_r - new_r) / temp)
            if new_r >= current_r:
                centers = new_centers
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
            else:
                delta = current_r - new_r
                if delta < 1e-9:
                    # Numerical noise, accept
                    centers = new_centers
                    current_r = new_r
                else:
                    prob = math.exp(-delta / temp)
                    if np.random.random() < prob:
                        centers = new_centers
                        current_r = new_r
            
            temp *= cooling_rate
            
            # Occasionally, if stuck, do a larger random restart of a few points?
            # Or just let cooling handle it.
            
        return best_centers, best_r

    # 1. Initialize
    centers = initialize_hex_grid()
    
    # 2. Relax using SA
    refined_centers, refined_r = simulated_annealing(centers, max_time=135.0)
    
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
