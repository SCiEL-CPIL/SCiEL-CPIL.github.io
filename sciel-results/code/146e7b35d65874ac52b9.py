"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Improved Simulated Annealing for packing 50 circles in a unit square.
    
    Key improvements:
    1. Better initialization using a randomized hexagonal grid.
    2. Efficient score calculation using NumPy vectorization.
    3. Adaptive step size and reheating mechanism to escape local minima.
    4. Time-based termination to maximize performance within the limit.
    """
    np.random.seed(42)
    
    # --- Initialization ---
    def hex_init():
        centers = np.zeros((N, 2))
        # Hexagonal spacing estimate
        r_est = 0.07
        dx = 2 * r_est
        dy = dx * np.sqrt(3) / 2
        
        # Generate points in a larger area
        xs = np.arange(-1.0, 1.0, dx)
        ys = np.arange(-1.0, 1.0, dy)
        
        points = []
        for i, y in enumerate(ys):
            offset = 0 if i % 2 == 0 else dx / 2
            for x in xs:
                points.append([x + offset, y])
                
        points = np.array(points)
        
        # Keep only points inside the square [-0.5, 0.5] with margin
        mask = (np.abs(points[:, 0]) <= 0.5 - r_est) & (np.abs(points[:, 1]) <= 0.5 - r_est)
        points = points[mask]
        
        # If we have more than N, select N randomly to break symmetry
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            centers = points[indices]
        else:
            # Pad if necessary
            centers = np.vstack([points, np.random.uniform(-0.4, 0.4, (N - len(points), 2))])
            
        return centers

    centers = hex_init()
    
    # --- Score Function ---
    def compute_score(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        min_wall = np.min(0.5 - np.abs(c))
        
        return min(min_pair, min_wall)

    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # --- Simulated Annealing ---
    T = 0.01
    T_min = 1e-9
    cooling = 0.9995
    
    # Adaptive step size
    step_size = 0.02
    
    # Counters
    no_improve = 0
    max_no_improve = 5000
    
    # Time limit check
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        # Pick random index
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Perturb
        # Use a combination of Gaussian and occasional large jumps
        if np.random.rand() < 0.1:
            # Large jump
            centers[i] = np.random.uniform(-0.49, 0.49, 2)
        else:
            # Small jump
            centers[i] += np.random.normal(0, step_size, 2)
            
        # Clip
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability
            delta = new_score - current_score
            # Avoid overflow/underflow in exp
            if delta / T > -700: 
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            
            if np.random.rand() < prob:
                current_score = new_score
                no_improve = 0
            else:
                centers[i] = old_pos
                no_improve += 1
        
        # Cooling
        T *= cooling
        
        # Adaptive step size
        if no_improve > max_no_improve:
            # Reheat and increase step size
            T = max(T, 0.005)
            step_size *= 1.5
            no_improve = 0
        elif no_improve > max_no_improve // 2:
            step_size *= 1.1
        else:
            step_size *= 0.99
        
        # Ensure step size doesn't get too small or large
        step_size = np.clip(step_size, 0.001, 0.1)

    return best_centers
# EVOLVE-BLOCK-END
