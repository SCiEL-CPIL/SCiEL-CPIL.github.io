"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for overhead/return
    
    # 1. Initialize with a grid layout
    # A 7x7 grid has 49 points. We need 50.
    # Let's use a 7x7 grid and add one point, or use a different grid.
    # Actually, 50 points in a 7x7 grid: 49 spots + 1 extra?
    # Better: Use a grid that fits 50 points reasonably well.
    # sqrt(50) ~ 7.07.
    # Let's try a 7x7 grid, but we have 50 points.
    # If we use 7x7, we have 49 cells. One cell must contain 2 points? No, they would overlap.
    # We can use a 6x9 grid? 54 cells.
    # Or just start with random points and let SA find the structure.
    # Starting from a grid is usually better for convergence speed.
    # Let's create a 7x7 grid of centers, but we need 50.
    # We can take the first 50 points from a larger grid, e.g., 8x8 (64 points).
    # Or, we can just use random initialization with a high temperature start.
    # Given the time budget, random init + SA is robust.
    
    # Let's try a hybrid: start with a grid of 7x7 (49 pts) and add a 50th at a random valid spot,
    # or simply use random initialization.
    # Random initialization is safer to avoid local minima of the specific grid.
    
    np.random.seed(42) # For reproducibility in local testing, but evaluator might not care. 
    # Actually, using a fixed seed is good for deterministic behavior if needed, 
    # but random is fine too. Let's use a fixed seed for stability.
    
    # Initial positions: Random inside the square
    # Ensure they are strictly inside to avoid immediate invalid state if we check boundaries strictly
    # But the score function handles boundaries.
    centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    # Calculate initial score
    def calc_score(centers):
        # Pairwise distances
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, np.inf)
        # min_pair_dist = np.min(dists)
        
        # Vectorized way without full NxN matrix if N is large, but N=50 is small.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        # dist to left: x + 0.5
        # dist to right: 0.5 - x
        # dist to bottom: y + 0.5
        # dist to top: 0.5 - y
        # min_dist_boundary = np.min(np.min(0.5 - np.abs(centers), axis=1))
        # Actually, for each point i:
        # d_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # r_boundary = min(d_i)
        
        dists_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dists_boundary)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    current_centers = centers.copy()
    current_score = calc_score(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-7
    cooling_rate = 0.99995
    
    # We will iterate until time runs out
    # Each step: pick a random index, perturb it, calculate new score
    
    while time.time() < deadline:
        # Perturb a random center
        idx = np.random.randint(N)
        # Step size: starts large, decreases with T
        step_size = T * 2.0
        delta = np.random.normal(0, step_size, 2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        
        # Ensure new center is within bounds? 
        # The score function penalizes going out of bounds (r becomes negative or small).
        # It's better to allow it and let the score guide it back, or clip.
        # Clipping might prevent finding better spots near the edge if the gradient is flat.
        # However, if a point goes far outside, r becomes very negative.
        # Let's just calculate score.
        
        new_score = calc_score(new_centers)
        
        # Acceptance criterion
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / T):
            current_centers = new_centers
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart with slightly higher T if stuck? 
            # For now, just keep cooling.
            
    return best_centers
# EVOLVE-BLOCK-END
