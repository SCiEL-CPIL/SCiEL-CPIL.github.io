"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-free optimizer using Particle Swarm Optimization (PSO)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap func to count evaluations
        eval_count = [0]
        def wrapped_func(x):
            eval_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        # PSO Parameters
        pop_size = 10
        w = 0.5  # Inertia
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social
        
        # Initialize particles
        positions = rng.uniform(-5, 5, (pop_size, dim))
        velocities = rng.uniform(-1, 1, (pop_size, dim))
        
        personal_best_pos = np.copy(positions)
        personal_best_val = np.full(pop_size, np.inf)
        global_best_pos = np.full(dim, np.inf)
        global_best_val = np.inf

        # Initial evaluation
        for i in range(pop_size):
            if eval_count[0] >= budget:
                break
            fx = wrapped_func(positions[i])
            personal_best_val[i] = fx
            personal_best_pos[i] = positions[i].copy()
            if fx < global_best_val:
                global_best_val = fx
                global_best_pos = positions[i].copy()

        # Main loop
        while eval_count[0] < budget:
            # Update velocities and positions
            for i in range(pop_size):
                if eval_count[0] >= budget:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_pos[i] - positions[i]) + c2 * r2 * (global_best_pos - positions[i])
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], -5.0, 5.0)
                
                # Evaluate
                fx = wrapped_func(positions[i])
                
                # Update personal best
                if fx < personal_best_val[i]:
                    personal_best_val[i] = fx
                    personal_best_pos[i] = positions[i].copy()
                    
                    # Update global best
                    if fx < global_best_val:
                        global_best_val = fx
                        global_best_pos = positions[i].copy()

        return global_best_pos, global_best_val
# EVOLVE-BLOCK-END
