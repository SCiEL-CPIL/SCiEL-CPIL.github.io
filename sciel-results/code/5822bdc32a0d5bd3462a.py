"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization (PSO) with time-bounded execution."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        
        dim = self.dim
        budget_remaining = self.budget
        
        # PSO Parameters
        swarm_size = 20  # Number of particles
        w = 0.7          # Inertia weight
        c1 = 1.5         # Cognitive coefficient
        c2 = 1.5         # Social coefficient
        
        lower = -5.0
        upper = 5.0
        
        # Initialize swarm
        # Use a fixed seed for reproducibility in local testing, but the evaluator
        # creates a new instance. Using a fixed seed is fine as the function is fixed.
        rng = np.random.default_rng(42)
        
        # Initialize positions randomly in the box
        positions = rng.uniform(lower, upper, (swarm_size, dim))
        velocities = rng.uniform(-1, 1, (swarm_size, dim))
        
        # Evaluate initial positions
        fitness = np.array([func(pos) for pos in positions])
        budget_remaining -= swarm_size
        
        # Update personal bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        
        # Update global best
        global_best_idx = np.argmin(personal_best_fit)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = personal_best_fit[global_best_idx]
        
        # Main loop
        while budget_remaining > 0:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine how many particles we can update this iteration
            # We update all particles in one iteration, costing swarm_size evaluations
            if budget_remaining < swarm_size:
                # If we can't afford a full swarm update, we can still do some individual updates
                # but for simplicity and to maintain PSO dynamics, we might stop or do partial.
                # Let's just stop if we can't do a full iteration to keep code simple and robust.
                break
            
            # Update velocities and positions
            for i in range(swarm_size):
                # r1, r2 are random numbers in [0, 1]
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Velocity update
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (personal_best_pos[i] - positions[i]) + 
                                 c2 * r2 * (global_best_pos - positions[i]))
                
                # Position update
                new_pos = positions[i] + velocities[i]
                
                # Clip to bounds
                new_pos = np.clip(new_pos, lower, upper)
                
                # Evaluate
                new_fit = func(new_pos)
                budget_remaining -= 1
                
                # Update personal best
                if new_fit < personal_best_fit[i]:
                    personal_best_fit[i] = new_fit
                    personal_best_pos[i] = new_pos.copy()
                    
                    # Update global best
                    if new_fit < global_best_fit:
                        global_best_fit = new_fit
                        global_best_pos = new_pos.copy()
                        
                # Update current position
                positions[i] = new_pos
                
                # Check time inside the loop for tight control
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
                
        return global_best_pos, global_best_fit
# EVOLVE-BLOCK-END
