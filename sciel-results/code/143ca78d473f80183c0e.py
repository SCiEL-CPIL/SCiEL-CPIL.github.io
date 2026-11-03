"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Pre-compute indices for pair checks
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def is_valid(pos, rad):
        # Check boundaries
        if np.any(pos[:, 0] - rad < -1e-10) or np.any(pos[:, 0] + rad > 1 + 1e-10):
            return False
        if np.any(pos[:, 1] - rad < -1e-10) or np.any(pos[:, 1] + rad > 1 + 1e-10):
            return False
        
        # Check overlaps
        dx = pos[i_indices, 0] - pos[j_indices, 0]
        dy = pos[i_indices, 1] - pos[j_indices, 1]
        dist_sq = dx*dx + dy*dy
        rad_sum = rad[i_indices] + rad[j_indices]
        if np.any(dist_sq < rad_sum**2 - 1e-10):
            return False
        return True
    
    def initial_placement(seed):
        np.random.seed(seed)
        # Use a hexagonal-like grid for better initial packing
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Try different grid configurations
        for side in [5, 6]:
            if side * side >= N:
                break
        
        # Place on a grid with some jitter
        for i in range(N):
            row = i // side
            col = i % side
            # Add offset for hexagonal pattern
            offset = 0.5 / side if row % 2 == 1 else 0
            positions[i, 0] = (col + offset + 0.5) / side
            positions[i, 1] = (row + 0.5) / side
            radii[i] = 0.3 / side
        
        # Small random perturbation
        positions += np.random.uniform(-0.01, 0.01, (N, 2))
        
        # Clip to valid range
        positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
        positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)
        
        return positions, radii
    
    def simulated_annealing(init_pos, init_rad, time_budget, seed):
        np.random.seed(seed)
        current_pos = init_pos.copy()
        current_rad = init_rad.copy()
        
        if not is_valid(current_pos, current_rad):
            return None, None, -np.inf
        
        current_score = np.sum(current_rad)
        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score
        
        # Adaptive parameters
        T = 0.05
        T_min = 0.0005
        alpha = 0.9998
        step_size_pos = 0.03
        step_size_rad = 0.03
        start_time = time.time()
        
        for it in range(10000000):
            if time.time() - start_time > time_budget:
                break
            
            # Generate a move
            i = np.random.randint(N)
            move_type = np.random.rand()
            
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            
            if move_type < 0.6:
                # Move position
                delta = np.random.normal(0, step_size_pos, 2)
                new_pos[i] += delta
            else:
                # Change radius
                delta_r = np.random.normal(0, step_size_rad)
                new_rad[i] += delta_r
                new_rad[i] = np.clip(new_rad[i], 0.005, 0.5)
            
            # Check validity
            if is_valid(new_pos, new_rad):
                new_score = np.sum(new_rad)
                delta_score = new_score - current_score
                
                # Acceptance probability
                if delta_score > 0:
                    accept = True
                else:
                    if T > T_min:
                        accept_prob = np.exp(delta_score / T)
                        accept = np.random.rand() < accept_prob
                    else:
                        accept = False
                
                if accept:
                    current_pos = new_pos
                    current_rad = new_rad
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_pos = current_pos.copy()
                        best_rad = current_rad.copy()
            else:
                # If invalid, slightly reduce step size
                step_size_pos *= 0.999
                step_size_rad *= 0.999
            
            # Cool down and adapt step sizes
            T *= alpha
            if T < T_min:
                T = T_min
            step_size_pos = max(0.001, step_size_pos * 0.99999)
            step_size_rad = max(0.001, step_size_rad * 0.99999)
        
        return best_pos, best_rad, best_score
    
    # Run multiple SA runs with different seeds
    start_time = time.time()
    time_limit = 108  # Keep buffer for 120s limit
    
    best_overall_pos = None
    best_overall_rad = None
    best_overall_score = -np.inf
    
    num_runs = 5
    time_per_run = time_limit / num_runs
    
    for run in range(num_runs):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            break
        
        seed = 42 + run * 1000
        init_pos, init_rad = initial_placement(seed)
        
        pos, rad, score = simulated_annealing(init_pos, init_rad, min(time_per_run, remaining_time), seed)
        
        if pos is not None and score > best_overall_score:
            best_overall_score = score
            best_overall_pos = pos
            best_overall_rad = rad
    
    # Final validation
    if best_overall_pos is None or not is_valid(best_overall_pos, best_overall_rad):
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.3 / side])
        return np.array(cells)
    
    return np.column_stack([best_overall_pos, best_overall_rad])
# EVOLVE-BLOCK-END
