"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def compute_radii(centers):
        """
        Compute maximum radii for given centers.
        centers: (N, 2) array
        returns: (N,) array of radii
        """
        n = len(centers)
        # Boundary distances
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        boundary_dists = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Pairwise distances
        # dist[i, j] = ||centers[i] - centers[j]||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # For each i, the max radius is min(boundary_dists[i], min_j dist[i, j] / 2)
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dist, np.inf)
        pair_dists_min = np.min(dist, axis=1) / 2.0
        
        radii = np.minimum(boundary_dists, pair_dists_min)
        return radii

    def objective(centers):
        """
        Objective: Sum of radii. Higher is better.
        """
        radii = compute_radii(centers)
        return np.sum(radii)

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        # Start in a slightly inset area to avoid boundary issues initially
        centers = rng.uniform(0.05, 0.95, size=(N, 2))
        return centers

    best_score = -np.inf
    best_centers = None
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-5
    n_steps = 80000
    n_restarts = 5
    
    # Pre-allocate for speed? Not strictly necessary for N=26, but good practice.
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        if restart == 0:
            centers = initial_grid_centers()
            # Add small jitter to break symmetry
            centers += np.random.normal(0, 0.01, size=centers.shape)
        else:
            # Use best found so far + noise, or pure random
            if best_centers is not None and restart < 3:
                centers = best_centers + np.random.normal(0, 0.05, size=best_centers.shape)
            else:
                centers = random_centers(seed=restart)
            
        centers = np.clip(centers, 0.0, 1.0)
        
        current_score = objective(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Annealing Loop
        # Use a geometric cooling schedule
        cooling_factor = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start
        
        # Local best tracking for this run
        local_best_score = current_score
        local_best_centers = centers.copy()
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
                
            # Generate move
            # Pick a random center to move
            idx = np.random.randint(N)
            # Gaussian move
            # Step size can be adaptive or fixed. Fixed relative to domain size.
            move_size = 0.05 * (T / T_start) + 0.001 # Smaller moves as T drops
            new_centers = centers.copy()
            new_centers[idx] += np.random.normal(0, move_size, size=2)
            new_centers[idx] = np.clip(new_centers[idx], 0.0, 1.0)
            
            new_score = objective(new_centers)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                centers = new_centers
                current_score = new_score
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_centers = centers.copy()
                    
            # Cool down
            T *= cooling_factor
            
        # Update global best
        if local_best_score > best_score:
            best_score = local_best_score
            best_centers = local_best_centers.copy()

    # Final Polish: Gradient-free local refinement (Nelder-Mead or similar) 
    # or just a few more SA steps with very low T.
    # Given time constraints, we'll do a high-precision SA pass on the best center.
    if best_centers is not None and time.time() < deadline - 2:
        centers = best_centers.copy()
        current_score = objective(centers)
        
        T = 1e-4
        cooling_factor = (1e-6 / T) ** (1.0 / 5000)
        
        for step in range(5000):
            if time.time() > deadline - 1:
                break
            idx = np.random.randint(N)
            move_size = 0.001
            new_centers = centers.copy()
            new_centers[idx] += np.random.normal(0, move_size, size=2)
            new_centers[idx] = np.clip(new_centers[idx], 0.0, 1.0)
            
            new_score = objective(new_centers)
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                centers = new_centers
                current_score = new_score
            
            T *= cooling_factor
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()

    # Construct final result
    radii = compute_radii(best_centers)
    result = np.column_stack([best_centers, radii])
    
    # Final validation check (though radii are computed to be valid)
    # We return the array as requested
    return result

# EVOLVE-BLOCK-END
