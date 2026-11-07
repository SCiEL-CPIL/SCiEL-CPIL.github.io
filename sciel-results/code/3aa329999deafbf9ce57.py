"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """PSO optimizer for BBOB-style Schwefel function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # PSO parameters
        n_particles = 20
        w0 = 0.9      # Initial inertia weight
        w1 = 0.4      # Final inertia weight
        c1 = 2.0      # Cognitive coefficient
        c2 = 2.0      # Social coefficient
        
        # Initialize particles uniformly in [-5, 5]^dim
        particles = rng.uniform(-5, 5, size=(n_particles, dim))
        velocities = rng.uniform(-1, 1, size=(n_particles, dim))
        
        # Evaluate initial particles
        pbests = np.zeros((n_particles, dim))
        pbest_f = np.full(n_particles, np.inf)
        gbest = np.zeros(dim)
        gbest_f = np.inf
        
        evals = 0
        for i in range(n_particles):
            try:
                f = func(particles[i])
            except Exception:
                f = np.inf
            evals += 1
            
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbests[i] = particles[i].copy()
            
            if f < gbest_f:
                gbest_f = f
                gbest = particles[i].copy()
        
        # Total number of iterations
        max_iters = (budget - evals) // n_particles
        if max_iters < 1:
            return gbest, gbest_f
        
        # Main PSO loop
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Linearly decay inertia weight
            w = w0 + (w1 - w0) * (it / max(max_iters - 1, 1))
            
            for i in range(n_particles):
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (pbests[i] - particles[i]) + 
                                c2 * r2 * (gbest - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                np.clip(particles[i], -5, 5, out=particles[i])
                
                # Evaluate
                try:
                    f = func(particles[i])
                except Exception:
                    f = np.inf
                evals += 1
                
                # Update personal best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbests[i] = particles[i].copy()
                
                # Update global best
                if f < gbest_f:
                    gbest_f = f
                    gbest = particles[i].copy()
        
        return gbest, gbest_f
# EVOLVE-BLOCK-END
