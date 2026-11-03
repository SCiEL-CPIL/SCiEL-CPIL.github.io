import numpy as np
from scipy.stats import qmc
import time

N = 50

def compute_score(centers):
    """
    Computes the score for a given set of centers.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    # Centers are in [-0.5, 0.5]^3
    # 1. Distances to walls
    # min_k (0.5 - |c_ik|)
    # For each point i, dist to wall is min(0.5 - x_i, x_i + 0.5, ...) 
    # Since x_i in [-0.5, 0.5], dist to x=0.5 is 0.5-x_i, to x=-0.5 is x_i+0.5.
    # Min dist to any wall for point i is min(0.5 - abs(x_i), 0.5 - abs(y_i), 0.5 - abs(z_i))?
    # No, dist to plane x=0.5 is 0.5 - x_i. Dist to plane x=-0.5 is x_i + 0.5.
    # The minimum of these two is 0.5 - abs(x_i).
    # So min dist to walls for point i is min(0.5 - abs(c_i0), 0.5 - abs(c_i1), 0.5 - abs(c_i2)).
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    # 2. Pairwise distances
    # Use broadcasting
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists) / 2.0
    
    # More efficient for N=50
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_wall_dist, min_pair_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # 1. Initialization using Sobol sequence for low-discrepancy
    sobol_engine = qmc.Sobol(d=3, scramble=True, seed=42)
    # Generate 50 points in [0, 1]^3
    points = sobol_engine.random(50)
    # Scale to [-0.5, 0.5]^3
    centers = points * 1.0 - 0.5
    
    # Ensure they are strictly inside (Sobol points are in [0,1), so they are in [-0.5, 0.5))
    # But we need to ensure they don't touch the boundary if possible, 
    # though the score handles boundaries.
    
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # 2. Local Optimization
    # We will use a simple hill-climbing with random restarts/kicks.
    # We focus on the "bottleneck" constraints.
    
    def find_bottleneck(centers):
        """Returns the type of bottleneck and indices."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
        min_wall_val = np.min(wall_dists[min_wall_idx])
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        if min_wall_val < min_pair_val:
            return 'wall', min_wall_idx, min_wall_val
        else:
            return 'pair', min_pair_idx, min_pair_val

    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        bottleneck_type, indices, bottleneck_val = find_bottleneck(centers)
        
        # Try to move the involved point(s) to increase the bottleneck distance
        # For 'wall', move point 'indices' away from the wall
        # For 'pair', move point 'indices[0]' away from 'indices[1]'
        
        moved = False
        if bottleneck_type == 'wall':
            i = indices
            # Determine which wall is closest
            c = centers[i]
            # dist to x=0.5 is 0.5-c[0], to x=-0.5 is c[0]+0.5
            # We want to increase the min distance.
            # Move towards the center of the cube?
            # Gradient of (0.5 - abs(x)) is -sign(x).
            # To increase 0.5 - abs(x), we should move x towards 0.
            # So move in direction -sign(c).
            direction = -np.sign(c)
            # Add some randomness to escape local minima
            direction += np.random.normal(0, 0.1, 3)
            
            new_center = centers.copy()
            new_center[i] = c + step_size * direction
            # Clip to cube to ensure validity? 
            # If we clip, the score might drop if we hit the wall.
            # But the score function handles it.
            new_score = compute_score(new_center)
            
            if new_score > current_score:
                centers = new_center
                current_score = new_score
                moved = True
            else:
                # Try moving in the opposite direction? Or random direction?
                # Let's try a random perturbation
                new_center = centers.copy()
                new_center[i] = c + step_size * np.random.normal(0, 1, 3)
                new_score = compute_score(new_center)
                if new_score > current_score:
                    centers = new_center
                    current_score = new_score
                    moved = True
        
        elif bottleneck_type == 'pair':
            i, j = indices
            # Move i away from j
            direction = centers[i] - centers[j]
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            direction += np.random.normal(0, 0.1, 3)
            
            new_center = centers.copy()
            new_center[i] = centers[i] + step_size * direction
            new_score = compute_score(new_center)
            
            if new_score > current_score:
                centers = new_center
                current_score = new_score
                moved = True
            else:
                # Try moving j away from i
                new_center = centers.copy()
                new_center[j] = centers[j] - step_size * direction
                new_score = compute_score(new_center)
                if new_score > current_score:
                    centers = new_center
                    current_score = new_score
                    moved = True
                else:
                    # Random kick
                    new_center = centers.copy()
                    new_center[i] = centers[i] + step_size * np.random.normal(0, 1, 3)
                    new_score = compute_score(new_center)
                    if new_score > current_score:
                        centers = new_center
                        current_score = new_score
                        moved = True
        
        if moved:
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            no_improve_count += 1
            # If stuck, try a larger random kick or restart step
            if no_improve_count > max_no_improve:
                # Random kick to a random point
                idx = np.random.randint(N)
                centers[idx] += 0.05 * np.random.normal(0, 1, 3)
                # Clip to valid range to avoid huge penalties, though score handles it
                centers = np.clip(centers, -0.5, 0.5)
                current_score = compute_score(centers)
                no_improve_count = 0
                max_no_improve += 10 # Be more patient next time
    
    return best_centers

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
import time

N = 50

def compute_score(centers):
    """
    Computes the score for a given set of centers.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_wall_dist, min_pair_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # 1. Initialization using Sobol sequence for low-discrepancy
    sobol_engine = qmc.Sobol(d=3, scramble=True, seed=42)
    points = sobol_engine.random(50)
    centers = points * 1.0 - 0.5
    
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Find bottleneck
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
        min_wall_val = np.min(wall_dists[min_wall_idx])
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        moved = False
        if min_wall_val < min_pair_val:
            i = min_wall_idx
            c = centers[i]
            direction = -np.sign(c)
            direction += np.random.normal(0, 0.1, 3)
            
            new_center = centers.copy()
            new_center[i] = c + step_size * direction
            new_score = compute_score(new_center)
            
            if new_score > current_score:
                centers = new_center
                current_score = new_score
                moved = True
            else:
                new_center = centers.copy()
                new_center[i] = c + step_size * np.random.normal(0, 1, 3)
                new_score = compute_score(new_center)
                if new_score > current_score:
                    centers = new_center
                    current_score = new_score
                    moved = True
        else:
            i, j = min_pair_idx
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction = direction / norm
            direction += np.random.normal(0, 0.1, 3)
            
            new_center = centers.copy()
            new_center[i] = centers[i] + step_size * direction
            new_score = compute_score(new_center)
            
            if new_score > current_score:
                centers = new_center
                current_score = new_score
                moved = True
            else:
                new_center = centers.copy()
                new_center[j] = centers[j] - step_size * direction
                new_score = compute_score(new_center)
                if new_score > current_score:
                    centers = new_center
                    current_score = new_score
                    moved = True
                else:
                    new_center = centers.copy()
                    new_center[i] = centers[i] + step_size * np.random.normal(0, 1, 3)
                    new_score = compute_score(new_center)
                    if new_score > current_score:
                        centers = new_center
                        current_score = new_score
                        moved = True
        
        if moved:
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                idx = np.random.randint(N)
                centers[idx] += 0.05 * np.random.normal(0, 1, 3)
                centers = np.clip(centers, -0.5, 0.5)
                current_score = compute_score(centers)
                no_improve_count = 0
                max_no_improve += 10
    
    return best_centers
# EVOLVE-BLOCK-END
