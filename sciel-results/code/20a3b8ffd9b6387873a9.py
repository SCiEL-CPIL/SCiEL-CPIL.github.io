"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) with adaptive inertia."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        range_bounds = upper - lower

        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            return func(x)

        # PSO Parameters
        # We have 500 calls. Let's use a population of 20.
        # That allows for 25 generations (20 * 25 = 500).
        # If we use 10, we get 50 generations.
        # Let's try 15 particles -> 33 generations.
        pop_size = 15
        max_iter = budget // pop_size
        
        # Initialize particles
        # Uniform random initialization
        particles = np.random.uniform(lower, upper, (pop_size, dim))
        velocities = np.random.uniform(-1.0, 1.0, (pop_size, dim)) * 0.1 # Small initial velocity
        
        pbests = particles.copy()
        pbest_f = np.full(pop_size, np.inf)
        gbest = particles[0].copy()
        gbest_f = np.inf
        
        # Cognitive and social coefficients
        c1 = 1.5
        c2 = 1.5
        # Inertia weight decay
        w_start = 0.9
        w_end = 0.4
        
        for t in range(max_iter):
            # Update inertia weight
            w = w_start - (w_start - w_end) * (t / max_iter)
            
            # Evaluate current positions
            for i in range(pop_size):
                if call_count[0] >= budget:
                    break
                f_val = counting_func(particles[i])
                if f_val < pbest_f[i]:
                    pbest_f[i] = f_val
                    pbests[i] = particles[i]
                if f_val < gbest_f:
                    gbest_f = f_val
                    gbest = particles[i]
            
            if call_count[0] >= budget:
                break

            # Update velocities and positions for next iteration
            for i in range(pop_size):
                if call_count[0] >= budget - pop_size: # Ensure we have budget for next evals if we update
                     # Actually, we evaluate in the next loop. 
                     # We need to ensure we don't exceed budget during evaluation.
                     pass
                 
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbests[i] - particles[i]) + 
                                 c2 * r2 * (gbest - particles[i]))
                
                # Clip velocities to prevent large jumps
                # A common heuristic is to limit velocity to a fraction of the range
                v_max = 0.5 * range_bounds
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                particles[i] = particles[i] + velocities[i]
                
                # Reflect or clip at boundaries
                # Clipping is simpler and robust
                particles[i] = np.clip(particles[i], lower, upper)

        return gbest, gbest_f
# EVOLVE-BLOCK-END
