"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: PSO + Nelder-Mead polish for rotated Griewank-Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: PSO (Particle Swarm Optimization)
        # PSO is well-suited for rotated, non-convex landscapes like Griewank-Rosenbrock
        # because it uses social and cognitive memory without gradient information.
        
        n_particles = 20
        max_iter = 30
        
        # Initialize particles uniformly in [-4, 4]^dim
        particles = rng.uniform(-4, 4, (n_particles, self.dim))
        velocities = rng.uniform(-1, 1, (n_particles, self.dim))
        
        # Evaluate initial particles
        p_best = np.zeros((n_particles, self.dim))
        p_best_f = np.full(n_particles, np.inf)
        g_best = self.best_x.copy()
        g_best_f = self.best_f
        
        for i in range(n_particles):
            if self.calls >= self.budget or time.time() > self.deadline - 30:
                break
            f_i = wrapped_func(particles[i])
            if f_i < p_best_f[i]:
                p_best_f[i] = f_i
                p_best[i] = particles[i].copy()
            if f_i < g_best_f:
                g_best_f = f_i
                g_best = particles[i].copy()
        
        # PSO parameters with constriction factor
        # w: inertia weight, c1: cognitive, c2: social
        w_init = 0.9
        w_end = 0.4
        c1 = 1.49445
        c2 = 1.49445
        
        for it in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline - 40:
                break
            
            # Linearly decrease inertia weight
            w = w_init + (w_end - w_init) * (it / max_iter)
            
            for i in range(n_particles):
                if self.calls >= self.budget or time.time() > self.deadline - 40:
                    break
                
                # Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (p_best[i] - particles[i]) + 
                                c2 * r2 * (g_best - particles[i]))
                
                # Clip velocity to prevent explosion
                velocities[i] = np.clip(velocities[i], -2, 2)
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], -5, 5)
                
                # Evaluate new position
                f_new = wrapped_func(particles[i])
                if f_new < p_best_f[i]:
                    p_best_f[i] = f_new
                    p_best[i] = particles[i].copy()
                if f_new < g_best_f:
                    g_best_f = f_new
                    g_best = particles[i].copy()
        
        # Update global best from PSO
        if g_best_f < self.best_f:
            self.best_f = g_best_f
            self.best_x = g_best.copy()

        # Phase 2: Nelder-Mead polish on the best result
        if self.calls < self.budget and time.time() < self.deadline - 15:
            try:
                # Create a small initial simplex around best_x
                simplex = [self.best_x.copy()]
                for i in range(self.dim):
                    v = self.best_x.copy()
                    step = 0.05
                    v[i] += step
                    simplex.append(v)
                
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12,
                                        'initial_simplex': np.array(simplex),
                                        'adaptive': True})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final local random search with decreasing step size
        if self.calls < self.budget and time.time() < self.deadline - 5:
            remaining = min(self.budget - self.calls, 150)
            for i in range(remaining):
                if time.time() > self.deadline - 5:
                    break
                step = 0.2 * (0.7 ** (i // 20))
                x_try = self.best_x + rng.normal(0, step, self.dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = wrapped_func(x_try)
                # wrapped_func already updates best if improved

        # Ensure we have a valid result
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
