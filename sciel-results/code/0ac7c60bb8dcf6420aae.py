import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Particle Swarm Optimization (PSO)
        n_particles = 20
        w = 0.7  # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        # Initialize particles
        X = rng.uniform(-5, 5, (n_particles, dim))
        V = rng.uniform(-1, 1, (n_particles, dim))
        
        # Evaluate initial population
        F = np.array([safe_func(p) for p in X])
        
        # Personal bests
        P_best = X.copy()
        P_best_f = F.copy()
        
        # Global best
        global_idx = np.argmin(F)
        G_best = X[global_idx].copy()
        G_best_f = F[global_idx]
        
        if G_best_f < best_f:
            best_f = G_best_f
            best_x = G_best.copy()

        # PSO Iterations
        # Each iteration costs n_particles evaluations
        # We have roughly budget // n_particles iterations
        max_iters = budget // n_particles
        
        for it in range(max_iters):
            if not time_left() or call_count >= budget - n_particles:
                break
            
            # Update velocities
            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
            
            # Limit velocity to prevent oscillation
            max_v = 2.0
            V = np.clip(V, -max_v, max_v)
            
            # Update positions
            X_new = X + V
            X_new = np.clip(X_new, -5, 5)
            
            # Evaluate new positions
            F_new = np.array([safe_func(x) for x in X_new])
            
            # Update personal bests
            improved = F_new < P_best_f
            P_best[improved] = X_new[improved]
            P_best_f[improved] = F_new[improved]
            
            # Update global best
            min_idx = np.argmin(F_new)
            if F_new[min_idx] < G_best_f:
                G_best_f = F_new[min_idx]
                G_best = X_new[min_idx].copy()
                
                if G_best_f < best_f:
                    best_f = G_best_f
                    best_x = G_best.copy()
            
            X = X_new
            F = F_new

        # Phase 2: Local Refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - 1 - call_count
            
            for _ in range(remaining):
                if not time_left() or call_count >= budget - 1:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                sigma = np.clip(sigma, 0.001, 2.0)

        return best_x, best_f
