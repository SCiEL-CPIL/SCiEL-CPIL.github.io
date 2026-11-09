"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        # Ensure n_particles doesn't exceed budget
        n_particles = min(20, max(4, budget // 10))
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        
        # Initialize particles
        x = rng.uniform(-5, 5, (n_particles, dim))
        v = rng.uniform(-1, 1, (n_particles, dim))
        
        pbest_x = x.copy()
        pbest_f = np.full(n_particles, np.inf)
        
        gbest_x = np.zeros(dim)
        gbest_f = np.inf
        
        calls_made = 0
        
        def count_call(x_val):
            nonlocal calls_made
            calls_made += 1
            return func(x_val)

        # Evaluate initial population
        for i in range(n_particles):
            if calls_made >= budget:
                break
            f_i = count_call(x[i])
            pbest_f[i] = f_i
            if f_i < gbest_f:
                gbest_f = f_i
                gbest_x = x[i].copy()
        
        # If we ran out of budget during initialization, return best
        if calls_made >= budget:
            return gbest_x, gbest_f

        # Main loop
        # Each iteration consumes n_particles calls
        max_iters = (budget - calls_made) // n_particles
        
        for it in range(max_iters):
            # Update inertia weight
            w = w_start - (w_start - w_end) * (it / max(max_iters - 1, 1))
            
            for i in range(n_particles):
                if calls_made >= budget:
                    break
                
                # Generate random numbers
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Update velocity
                v[i] = w * v[i] + c1 * r1 * (pbest_x[i] - x[i]) + c2 * r2 * (gbest_x - x[i])
                
                # Update position
                x[i] = x[i] + v[i]
                
                # Clip to bounds
                x[i] = np.clip(x[i], -5, 5)
                
                # Evaluate
                f_i = count_call(x[i])
                
                # Update personal best
                if f_i < pbest_f[i]:
                    pbest_f[i] = f_i
                    pbest_x[i] = x[i].copy()
                    
                    # Update global best
                    if f_i < gbest_f:
                        gbest_f = f_i
                        gbest_x = x[i].copy()
                        
            if calls_made >= budget:
                break

        return gbest_x, gbest_f
# EVOLVE-BLOCK-END
