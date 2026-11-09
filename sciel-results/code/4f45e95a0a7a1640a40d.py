"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage derivative-free optimizer:
    1. Random sampling for initial exploration.
    2. Particle Swarm Optimization (PSO) for population-based search.
    3. Nelder-Mead local refinements from diverse starting points.
    4. (1+1)-ES polish with adaptive step size.
    
    Uses a time guard to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return (time.time() - start_time) > (deadline - 5.0)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Random sampling for broad exploration
        n_random = min(30, self.budget)
        for _ in range(n_random):
            if calls_used >= self.budget or check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_used += 1
            try_update(x, f)

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 2: Particle Swarm Optimization (PSO)
        # PSO with inertia, cognitive and social components
        n_particles = 15
        max_pso_iters = 200
        
        # Initialize particles around the best point found
        particles = np.zeros((n_particles, self.dim))
        pbest_x = np.zeros((n_particles, self.dim))
        pbest_f = np.full(n_particles, np.inf)
        
        # Initialize first particle at best found
        particles[0] = best_x.copy()
        pbest_x[0] = best_x.copy()
        pbest_f[0] = best_f
        
        # Initialize other particles with random positions
        for i in range(1, n_particles):
            if calls_used >= self.budget or check_time():
                break
            # Mix of random and perturbed best
            if rng.random() < 0.5:
                x = rng.uniform(-5, 5, self.dim)
            else:
                x = np.clip(best_x + rng.normal(0, 2.0, self.dim), -5, 5)
            f = func(x)
            calls_used += 1
            particles[i] = x
            pbest_x[i] = x
            pbest_f[i] = f
            try_update(x, f)

        # PSO parameters
        w = 0.729  # inertia weight
        c1 = 1.49445  # cognitive coefficient
        c2 = 1.49445  # social coefficient
        
        gbest_x = best_x.copy()
        gbest_f = best_f
        
        # Velocity initialization
        velocities = np.zeros((n_particles, self.dim))
        for i in range(n_particles):
            velocities[i] = rng.uniform(-1, 1, self.dim)

        for pso_iter in range(max_pso_iters):
            if calls_used >= self.budget or check_time():
                break
            
            for i in range(n_particles):
                if calls_used >= self.budget or check_time():
                    break
                
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (pbest_x[i] - particles[i]) + 
                                c2 * r2 * (gbest_x - particles[i]))
                
                # Limit velocity
                max_vel = 2.0
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], -5, 5)
                
                # Evaluate
                f = func(particles[i])
                calls_used += 1
                
                # Update personal best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = particles[i].copy()
                
                # Update global best
                if f < gbest_f:
                    gbest_f = f
                    gbest_x = particles[i].copy()
                    try_update(gbest_x, gbest_f)

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 3: Nelder-Mead local refinements
        n_nm = min(5, (self.budget - calls_used) // 50)
        
        # Select diverse starting points
        start_points = [best_x.copy()]
        for _ in range(min(n_nm - 1, 4)):
            if calls_used >= self.budget or check_time():
                break
            perturbation = rng.normal(0, 1.0, self.dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)

        for x_start in start_points:
            if calls_used >= self.budget or check_time():
                break
            
            local_calls = [0]
            
            def nm_func(x):
                nonlocal calls_used, local_calls
                if local_calls[0] >= 35 or calls_used >= self.budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls_used += 1
                local_calls[0] += 1
                return f

            try:
                result = minimize(
                    nm_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                if result.fun < best_f:
                    try_update(result.x, result.fun)
            except Exception:
                pass

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 4: (1+1)-ES polish
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        
        while calls_used < self.budget and not check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                sigma *= 1.2
            else:
                sigma *= 0.85
            
            if sigma < 1e-7:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
