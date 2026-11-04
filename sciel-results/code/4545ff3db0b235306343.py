"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    best_score = -np.inf
    best_config = None
    
    # Helper function to evaluate a configuration
    # Returns (sum_radii, penalty)
    # penalty is large if constraints are violated
    def evaluate(config):
        # config: (N, 3) array of [x, y, r]
        x = config[:, 0]
        y = config[:, 1]
        r = config[:, 2]
        
        # Boundary penalties
        # x must be in [r, 1-r], y in [r, 1-r]
        # Violation if x < r or x > 1-r or y < r or y > 1-r
        # We can compute the distance to the boundary for each circle
        # dist_to_boundary = min(x, y, 1-x, 1-y)
        # If r > dist_to_boundary, it's a violation.
        
        dists = np.minimum(np.minimum(x, y), np.minimum(1-x, 1-y))
        boundary_viol = np.maximum(0, r - dists)
        
        # Overlap penalties
        # For each pair i, j, distance d_ij must be >= r_i + r_j
        # We can compute this in a vectorized way using broadcasting
        # But N=26 is small, so O(N^2) is fine.
        # Let's use a loop for clarity and speed given small N, or vectorize.
        # Vectorization:
        # dx = x[:, None] - x[None, :]
        # dy = y[:, None] - y[None, :]
        # dist = np.sqrt(dx**2 + dy**2)
        # req_dist = r[:, None] + r[None, :]
        # overlap_viol = np.maximum(0, req_dist - dist)
        # Sum of overlaps (upper triangle only)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req_dist = r[:, None] + r[None, :]
        
        # Zero out diagonal and lower triangle to avoid double counting
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_viol = np.maximum(0, req_dist - dist)
        overlap_viol = np.where(mask, overlap_viol, 0)
        
        total_penalty = np.sum(boundary_viol) + np.sum(overlap_viol)
        
        sum_radii = np.sum(r)
        
        return sum_radii, total_penalty

    # Simulated Annealing function
    def simulated_annealing(initial_config, max_iters, initial_temp, final_temp, time_limit):
        config = initial_config.copy()
        current_sum, current_pen = evaluate(config)
        current_score = current_sum - 100.0 * current_pen
        
        best_local_sum = -np.inf
        best_local_config = None
        
        if current_pen < 1e-9:
            best_local_sum = current_sum
            best_local_config = config.copy()
            
        t_start = time.time()
        
        for i in range(max_iters):
            if time.time() > time_limit:
                break
                
            # Cooling schedule: linear or exponential
            # Let's use exponential cooling
            temp = initial_temp * (final_temp / initial_temp) ** (i / max_iters)
            
            # Generate a new configuration
            new_config = config.copy()
            
            # Randomly perturb some circles
            # 1. Move position
            if random.random() < 0.7:
                idx = random.randint(0, N-1)
                # Step size decreases with temperature
                step = temp * 0.1
                new_config[idx, 0] += random.uniform(-step, step)
                new_config[idx, 1] += random.uniform(-step, step)
                
            # 2. Change radius
            if random.random() < 0.3:
                idx = random.randint(0, N-1)
                step = temp * 0.1
                new_config[idx, 2] += random.uniform(-step, step)
                # Ensure radius is positive
                new_config[idx, 2] = max(1e-6, new_config[idx, 2])
            
            # 3. Global grow (try to increase all radii)
            if random.random() < 0.1:
                grow_step = temp * 0.05
                new_config[:, 2] += grow_step
                new_config[:, 2] = np.maximum(1e-6, new_config[:, 2])
            
            # Evaluate new configuration
            new_sum, new_pen = evaluate(new_config)
            new_score = new_sum - 100.0 * new_pen
            
            # Acceptance probability
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / temp):
                config = new_config
                current_sum = new_sum
                current_pen = new_pen
                current_score = new_score
                
                # Update best local if feasible
                if current_pen < 1e-9 and current_sum > best_local_sum:
                    best_local_sum = current_sum
                    best_local_config = config.copy()
                    
        return best_local_sum, best_local_config

    # Multi-start strategy
    # We will run several SA instances with different random seeds
    # We allocate time per run based on the deadline
    
    num_runs = 5
    time_per_run = (deadline - 5.0) / num_runs  # Leave 5s buffer for final check
    
    for run_id in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize configuration
        # Option 1: Grid
        # Option 2: Random
        if run_id % 2 == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.5 / side - 0.01
                        cells.append([x, y, r])
            config = np.array(cells)
        else:
            # Random initialization
            # Place circles randomly, ensuring they are small enough to fit
            # and try to avoid immediate massive overlaps by starting small
            config = np.zeros((N, 3))
            config[:, 0] = np.random.uniform(0.1, 0.9, N)
            config[:, 1] = np.random.uniform(0.1, 0.9, N)
            config[:, 2] = np.random.uniform(0.01, 0.05, N)
            
        # Run SA
        max_iters = 20000
        initial_temp = 0.5
        final_temp = 0.001
        
        local_best_sum, local_best_config = simulated_annealing(
            config, max_iters, initial_temp, final_temp, time.time() + time_per_run
        )
        
        if local_best_config is not None and local_best_sum > best_score:
            best_score = local_best_sum
            best_config = local_best_config

    # If no feasible solution found (should not happen with grid init), fallback to grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
        best_config = np.array(cells)
        best_score = np.sum(best_config[:, 2])
        
    # Final polish: try to squeeze out more by running a quick local optimization
    # or just return the best found. 
    # Given time constraints, returning the best SA result is usually sufficient.
    # We can do a simple "push" to maximize radii if there is slack.
    # But SA already tries to grow radii.
    
    return best_config

# EVOLVE-BLOCK-END
