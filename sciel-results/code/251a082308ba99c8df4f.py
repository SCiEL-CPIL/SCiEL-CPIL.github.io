"""PSO with adaptive restarts for Rastrigin rotated f15."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # PSO Parameters
        swarm_size = 15
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social
        w_start = 0.9
        w_end = 0.4
        v_max_ratio = 0.2  # Max velocity as fraction of range
        range_min = -4.5
        range_max = 4.5
        range_span = range_max - range_min
        v_max = v_max_ratio * range_span

        # Initialize Swarm
        # Use a mix of random and a structured start to ensure coverage
        positions = np.zeros((swarm_size, dim))
        velocities = np.zeros((swarm_size, dim))
        personal_bests = np.zeros((swarm_size, dim))
        personal_best_fits = np.full(swarm_size, np.inf)
        
        global_best_pos = np.zeros(dim)
        global_best_fit = np.inf

        calls_used = 0

        # Initial Evaluation
        for i in range(swarm_size):
            if time.time() > deadline or calls_used >= budget:
                break
            # Initialize with uniform random in [-4, 4]
            x = rng.uniform(-4.0, 4.0, dim)
            positions[i] = x
            fx = func(x)
            calls_used += 1
            personal_bests[i] = x
            personal_best_fits[i] = fx
            if fx < global_best_fit:
                global_best_fit = fx
                global_best_pos = x.copy()

        # PSO Loop
        # Estimate max iterations
        max_iters = (budget - calls_used) // swarm_size
        if max_iters < 5:
            max_iters = 5 # Minimum iterations to be useful

        no_improve_count = 0
        stagnation_limit = 5 # Iterations without global improvement

        for it in range(max_iters):
            if time.time() > deadline or calls_used >= budget:
                break
            
            # Update Inertia
            w = w_start - (w_start - w_end) * (it / max(max_iters - 1, 1))
            
            # Check for Stagnation and Restart if needed
            if no_improve_count >= stagnation_limit:
                # Restart: Re-initialize the worst half of the swarm
                # Sort indices by personal best fit
                sort_idx = np.argsort(personal_best_fits)
                worst_half = sort_idx[swarm_size // 2:]
                
                for idx in worst_half:
                    if time.time() > deadline or calls_used >= budget:
                        break
                    # Re-randomize around global best with large noise
                    noise = rng.normal(0, 2.0, dim)
                    new_x = global_best_pos + noise
                    new_x = np.clip(new_x, range_min, range_max)
                    
                    positions[idx] = new_x
                    velocities[idx] = rng.uniform(-v_max, v_max, dim)
                    
                    fx = func(new_x)
                    calls_used += 1
                    if fx < personal_best_fits[idx]:
                        personal_best_fits[idx] = fx
                        personal_bests[idx] = new_x
                    if fx < global_best_fit:
                        global_best_fit = fx
                        global_best_pos = new_x.copy()
                        no_improve_count = 0
                no_improve_count = 0 # Reset counter after restart

            # Update Swarm
            improved_global = False
            for i in range(swarm_size):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # Update Velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - positions[i]) + c2 * r2 * (global_best_pos - positions[i])
                
                # Clamp Velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update Position
                positions[i] += velocities[i]
                
                # Boundary Handling: Clip
                positions[i] = np.clip(positions[i], range_min, range_max)
                
                # Evaluate
                fx = func(positions[i])
                calls_used += 1
                
                # Update Personal Best
                if fx < personal_best_fits[i]:
                    personal_best_fits[i] = fx
                    personal_bests[i] = positions[i].copy()
                    
                    # Update Global Best
                    if fx < global_best_fit:
                        global_best_fit = fx
                        global_best_pos = positions[i].copy()
                        improved_global = True

            if improved_global:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return (global_best_pos, global_best_fit)
