import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0  # Safety margin
        
        # PSO Parameters
        swarm_size = 20
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max_ratio = 0.5
        
        # Initialize particles
        rng = np.random.default_rng(42)
        # Start with a spread out swarm to cover the box [-5, 5]
        positions = rng.uniform(-5, 5, (swarm_size, n))
        velocities = rng.uniform(-1, 1, (swarm_size, n))
        
        # Evaluate initial swarm
        fitnesses = np.array([func(pos) for pos in positions])
        evals = swarm_size
        
        pbest_positions = positions.copy()
        pbest_fitnesses = fitnesses.copy()
        
        gbest_idx = np.argmin(pbest_fitnesses)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitnesses[gbest_idx]
        
        best_x = gbest_position
        best_f = gbest_fitness
        
        # Pre-allocate for speed if possible, but list is fine for 500 evals
        
        while evals < budget:
            if time.time() - t_start > max_time:
                break
            
            # Calculate inertia weight decay
            # We don't know total generations in advance exactly, but we can estimate
            # Or just decay per generation.
            # Let's use a fixed decay per generation based on expected generations
            # Generations = (budget - swarm_size) / swarm_size
            # But we just loop until budget is hit.
            
            # Simple decay: w decreases linearly with number of generations
            # Let's just use a constant w or a simple decay based on evals
            # To be safe and simple, let's use a constant w=0.7 which is often good for small n
            # Or decay from 0.9 to 0.4 over the expected number of generations.
            # Expected gens = (500 - 20) / 20 = 24
            # Let's just update w based on the current generation count.
            
            # We need to track generation count
            # Since we don't have a gen counter in the loop structure easily, 
            # let's just use a constant w=0.7 for simplicity and robustness, 
            # or estimate it.
            
            # Actually, let's just do a simple decay based on evals ratio
            # This is a heuristic.
            progress = (evals - swarm_size) / max(1, (budget - swarm_size))
            w = w_start - (w_start - w_end) * min(1.0, progress)
            
            for i in range(swarm_size):
                if evals >= budget:
                    break
                
                r1 = rng.random()
                r2 = rng.random()
                
                velocities[i] = w * velocities[i] + \
                               c1 * r1 * (pbest_positions[i] - positions[i]) + \
                               c2 * r2 * (gbest_position - positions[i])
                
                # Clip velocity
                # v_max is relative to the box size? 
                # Box size is 10. v_max = 0.5 * 10 = 5?
                # Standard practice: v_max = v_max_ratio * (x_max - x_min)
                v_max = v_max_ratio * 10.0
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)
                
                f = func(positions[i])
                evals += 1
                
                if f < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = f
                    pbest_positions[i] = positions[i].copy()
                    
                    if f < gbest_fitness:
                        gbest_fitness = f
                        gbest_position = positions[i].copy()
                        best_x = positions[i].copy()
                        best_f = f
        
        return best_x, best_f
