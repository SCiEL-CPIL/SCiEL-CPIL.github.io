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
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Initialize particles
        X = rng.uniform(-5, 5, (n_particles, dim))
        V = rng.uniform(-1, 1, (n_particles, dim))
        
        # Evaluate initial population
        F_vals = np.array([safe_func(p) for p in X])
        
        # Personal bests
        pbest_X = X.copy()
        pbest_F = F_vals.copy()
        
        # Global best
        gbest_idx = np.argmin(pbest_F)
        gbest_X = pbest_X[gbest_idx].copy()
        gbest_F = pbest_F[gbest_idx]
        
        for i in range(n_particles):
            if F_vals[i] < best_f:
                best_f = F_vals[i]
                best_x = pbest_X[i].copy()

        # Run PSO
        while call_count < budget - 10 and time_left():
            for i in range(n_particles):
                if call_count >= budget - 5:
                    break
                
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], -5, 5)
                
                f_trial = safe_func(X[i])
                
                if f_trial < pbest_F[i]:
                    pbest_F[i] = f_trial
                    pbest_X[i] = X[i].copy()
                    
                    if f_trial < gbest_F:
                        gbest_F = f_trial
                        gbest_X = X[i].copy()
                        
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = X[i].copy()

        # Phase 2: Local Refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
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
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
