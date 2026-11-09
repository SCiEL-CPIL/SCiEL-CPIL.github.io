"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    # Create grid positions
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < N:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    
    # Initial radii: small enough to fit in grid cells
    # Cell width is 1/side. Max radius is 0.5/side. 
    # We start slightly smaller to allow growth.
    r_init = 0.45 / side
    radii = np.full(N, r_init)
    
    x = np.array(xs)
    y = np.array(ys)
    
    # Helper to check validity and calculate score
    def get_score(x, y, r):
        # Check boundaries
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return -np.inf
        # Check overlaps
        # dist^2 >= (r_i + r_j)^2
        # Vectorized pairwise distance check
        # To save time, we can skip full NxN if N is small, but 26 is small.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        r_sum_sq = (r[:, None] + r[None, :])**2
        # Diagonal is 0, so dist_sq < r_sum_sq (0 < (2r)^2) is true, but we ignore diagonal
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        # If any off-diagonal pair has dist_sq < r_sum_sq, it's invalid
        if np.any(dist_sq[mask] < r_sum_sq[mask] - 1e-12):
            return -np.inf
        return np.sum(r)

    current_score = get_score(x, y, radii)
    best_x = x.copy()
    best_y = y.copy()
    best_r = radii.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-5
    alpha = 0.995 # Cooling rate
    # We don't know exact steps, so we loop until time limit
    
    # Pre-compute indices for efficiency? No, vectorization is fast enough for N=26.
    
    while time.time() < deadline:
        # Choose a random circle
        i = np.random.randint(N)
        
        # Move type:
        # 1. Move position (x, y)
        # 2. Grow/Shrink radius
        if np.random.rand() < 0.7:
            # Move position
            # Step size proportional to radius? Or fixed?
            # Let's use a step size that allows exploration
            step = 0.01 * (1 + np.random.rand())
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            
            new_x = x.copy()
            new_y = y.copy()
            new_r = radii.copy()
            
            new_x[i] += dx
            new_y[i] += dy
            
            # Keep inside square (hard constraint for position move, adjust if needed? 
            # No, better to reject if out of bounds or overlap)
            if new_x[i] < new_r[i] or new_x[i] > 1 - new_r[i] or \
               new_y[i] < new_r[i] or new_y[i] > 1 - new_r[i]:
                # If it goes out of bounds, reject immediately to save time?
                # Or clamp? Clamping is a valid move if it stays valid.
                # Let's just reject for simplicity in SA, as "move" implies random perturbation.
                # However, clamping allows sliding along walls.
                # Let's try clamping for position moves.
                new_x[i] = np.clip(new_x[i], new_r[i], 1 - new_r[i])
                new_y[i] = np.clip(new_y[i], new_r[i], 1 - new_r[i])
                
                # Re-check overlap after clamp
                new_score = get_score(new_x, new_y, new_r)
            else:
                new_score = get_score(new_x, new_y, new_r)
                
        else:
            # Change radius
            # Grow or shrink
            if np.random.rand() < 0.6:
                # Try to grow
                delta_r = np.random.uniform(0, 0.005)
            else:
                # Shrink to allow others to move?
                delta_r = np.random.uniform(-0.005, 0)
            
            new_r = radii.copy()
            new_r[i] = np.clip(new_r[i] + delta_r, 1e-6, 0.5) # Max radius in unit square is 0.5
            
            new_x = x.copy()
            new_y = y.copy()
            new_score = get_score(new_x, new_y, new_r)
            
        # Acceptance criterion
        if new_score == -np.inf:
            # Invalid, reject
            # In standard SA with hard constraints, we just reject.
            pass
        else:
            if new_score > current_score:
                # Accept
                x = new_x
                y = new_y
                radii = new_r
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = radii.copy()
            else:
                # Worse, accept with probability exp((new - old) / T)
                # Since we maximize, delta is negative.
                delta = new_score - current_score
                if np.random.rand() < np.exp(delta / T):
                    x = new_x
                    y = new_y
                    radii = new_r
                    current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Optional: Reheat if stuck?
        # For now, keep it simple.
        
    # Return best found
    return np.column_stack((best_x, best_y, best_r))
# EVOLVE-BLOCK-END
