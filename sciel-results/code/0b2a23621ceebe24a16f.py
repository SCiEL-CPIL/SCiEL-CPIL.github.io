"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    def compute_radius(centers):
        """
        Compute the valid radius for a given set of centers.
        Returns (radius, bottleneck_info) where bottleneck_info helps identify 
        which constraint is limiting the radius.
        """
        # Wall constraint
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        
        # Pairwise constraint
        tree = cKDTree(centers)
        # k=2 gives self (dist 0) and nearest neighbor
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        
        # Determine bottleneck type for targeted moves
        if min_dist_to_wall <= min_pair_dist / 2.0:
            # Wall is the bottleneck
            # Find the point closest to the wall
            min_idx = np.argmin(0.5 - np.abs(centers))
            bottleneck_type = 'wall'
            bottleneck_idx = min_idx
            bottleneck_j = -1
        else:
            # Pair is the bottleneck
            # Find the pair with min distance
            min_dist = np.inf
            bottleneck_i = -1
            bottleneck_j = -1
            # cKDTree query returns indices
            # We need to find which pair had the min dist
            # Re-querying or storing is expensive. 
            # For N=200, we can just find the min pair from the k=2 results if we stored indices.
            # But tree.query(centers, k=2) returns distances and indices.
            # Let's re-do query to get indices for the min pair efficiently.
            # Actually, we can just use the min_dist found and find the pair.
            # Since N is small, we can just find the pair that achieves min_pair_dist.
            # But doing O(N^2) check every iteration is slow.
            # Let's use the tree query result properly.
            # The tree.query(centers, k=2) returns (dists, idxs)
            # dists shape (N, 2), idxs shape (N, 2)
            # dists[:, 0] is 0, idxs[:, 0] is self.
            # dists[:, 1] is dist to nearest, idxs[:, 1] is index of nearest.
            # The global min is min(dists[:, 1]).
            # Let's find which i has the global min.
            min_i = np.argmin(dists[:, 1])
            min_j = _idxs[min_i, 1] # Need to store idxs
            bottleneck_type = 'pair'
            bottleneck_idx = min_i
            bottleneck_j = min_j
            
        return r, (bottleneck_type, bottleneck_idx, bottleneck_j)

    # Helper to get radius and bottleneck with indices stored
    def compute_radius_with_info(centers):
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        
        if min_dist_to_wall <= min_pair_dist / 2.0:
            min_idx = np.argmin(0.5 - np.abs(centers))
            return r, 'wall', min_idx, -1
        else:
            min_i = np.argmin(dists[:, 1])
            min_j = idxs[min_i, 1]
            return r, 'pair', min_i, min_j

    def generate_hex_initial():
        """Generate a good initial hexagonal packing."""
        # Try to fit 200 circles in a hex grid
        # Approx 14x14 = 196, so 14x15 or similar
        best_pts = None
        best_r = 0
        
        for rows in range(12, 18):
            cols = math.ceil(N / rows)
            # Calculate max possible spacing for this grid
            # x spacing: 1/cols
            # y spacing: 1/((rows-1)*sqrt(3)/2 + 1)
            dx = 1.0 / cols
            dy = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1)
            d = min(dx, dy)
            
            pts = []
            for r in range(rows):
                y = -0.5 + d/2 + r * d * math.sqrt(3) / 2
                if r % 2 == 0:
                    x_start = -0.5 + d/2
                    n_in_row = cols
                else:
                    x_start = -0.5 + d
                    n_in_row = cols - 1
                for c in range(n_in_row):
                    x = x_start + c * d
                    pts.append([x, y])
            
            if len(pts) < N:
                # Pad with random points if not enough
                while len(pts) < N:
                    pts.append([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
            else:
                pts = pts[:N]
            
            pts = np.array(pts)
            r, _, _, _ = compute_radius_with_info(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
        
        return best_pts

    def sa_optimize(centers, max_time=120.0):
        """
        Simulated Annealing with bottleneck-targeted moves.
        """
        centers = centers.copy()
        current_r, curr_type, curr_i, curr_j = compute_radius_with_info(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        
        move_size = 0.05
        
        while time_left() > 5.0 and T > T_min:
            # Identify bottleneck
            if curr_type == 'pair':
                # Move one of the pair away from the other
                # Randomly pick which one to move, or move both?
                # Moving one is often more effective for escape.
                if np.random.random() < 0.5:
                    idx = curr_i
                    # Direction away from curr_j
                    diff = centers[idx] - centers[curr_j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        dir_vec = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                    else:
                        dir_vec = diff / norm
                else:
                    idx = curr_j
                    diff = centers[idx] - centers[curr_i]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        dir_vec = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                    else:
                        dir_vec = diff / norm
            else:
                # Wall bottleneck
                idx = curr_i
                # Direction away from the closest wall
                # Determine which wall
                if centers[idx, 0] > 0:
                    dir_vec = np.array([-1, 0])
                else:
                    dir_vec = np.array([1, 0])
                if centers[idx, 1] > 0:
                    # Check if y is closer
                    if abs(centers[idx, 1]) > abs(centers[idx, 0]):
                         dir_vec = np.array([0, -1])
                else:
                    if abs(centers[idx, 1]) > abs(centers[idx, 0]):
                         dir_vec = np.array([0, 1])
            
            # Add some randomness to the direction to allow exploration
            noise = np.random.randn(2) * 0.5
            dir_vec = dir_vec + noise
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)
            
            # Generate candidate move
            # Move size is scaled by temperature? Or fixed with random jitter?
            # Let's use a random step size proportional to T + base
            step = move_size * (0.5 + np.random.random())
            
            new_centers = centers.copy()
            new_centers[idx] += dir_vec * step
            
            # Keep inside square (hard constraint)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_r, new_type, new_i, new_j = compute_radius_with_info(new_centers)
            
            delta_r = new_r - current_r
            
            # Acceptance criterion
            if delta_r > 0 or np.random.random() < math.exp(delta_r / T):
                centers = new_centers
                current_r = new_r
                curr_type = new_type
                curr_i = new_i
                curr_j = new_j
                
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                    
            T *= cooling
            # Occasionally restart temperature if stuck?
            # Simple geometric cooling is fine for now.
            
        return best_centers, best_r

    # Main Strategy
    # 1. Generate initial hex
    initial_centers = generate_hex_initial()
    
    # 2. Run SA
    best_centers, best_r = sa_optimize(initial_centers)
    
    # 3. If time permits, try a few random restarts with SA
    # But SA is expensive. Let's just do one good run.
    # If we have time, do a short second run from the best result with higher T?
    if time_left() > 10.0:
        # Perturb best result slightly and run a short SA
        perturbed = best_centers + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Short SA run
        centers_copy = perturbed.copy()
        current_r, curr_type, curr_i, curr_j = compute_radius_with_info(centers_copy)
        T = 0.005
        T_min = 1e-6
        cooling = 0.99
        
        while time_left() > 5.0 and T > T_min:
            if curr_type == 'pair':
                idx = curr_i if np.random.random() < 0.5 else curr_j
                other = curr_j if idx == curr_i else curr_i
                diff = centers_copy[idx] - centers_copy[other]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    dir_vec = np.random.randn(2)
                else:
                    dir_vec = diff / norm
            else:
                idx = curr_i
                if abs(centers_copy[idx, 0]) > abs(centers_copy[idx, 1]):
                    dir_vec = np.array([-np.sign(centers_copy[idx, 0]), 0])
                else:
                    dir_vec = np.array([0, -np.sign(centers_copy[idx, 1])])
            
            noise = np.random.randn(2) * 0.5
            dir_vec = dir_vec + noise
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)
            
            step = 0.02 * (0.5 + np.random.random())
            
            new_centers = centers_copy.copy()
            new_centers[idx] += dir_vec * step
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_r, new_type, new_i, new_j = compute_radius_with_info(new_centers)
            delta_r = new_r - current_r
            
            if delta_r > 0 or np.random.random() < math.exp(delta_r / T):
                centers_copy = new_centers
                current_r = new_r
                curr_type = new_type
                curr_i = new_i
                curr_j = new_j
                
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers_copy.copy()
            
            T *= cooling

    return best_centers
# EVOLVE-BLOCK-END
