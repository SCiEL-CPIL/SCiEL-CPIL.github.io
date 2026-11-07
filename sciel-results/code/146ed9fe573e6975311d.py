"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization (PSO) for BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # PSO Parameters
        N = 25  # Population size
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w_start = 0.9
        w_end = 0.4
        v_max = 2.0
        
        # Initialize population
        # Positions uniformly in [-5, 5]^dim
        X = rng.uniform(-5, 5, (N, self.dim))
        # Velocities initially zero or small random
        V = rng.uniform(-1, 1, (N, self.dim))
        
        # Evaluate initial population
        F = np.array([func(x) for x in X])
        calls = N
        
        # Personal bests
        P_best = X.copy()
        F_p_best = F.copy()
        
        # Global best
        idx_min = np.argmin(F)
        G_best = X[idx_min].copy()
        F_g_best = F[idx_min]
        
        total_iters = self.budget - N
        if total_iters <= 0:
            return G_best, F_g_best
        
        for t in range(total_iters):
            if time.time() > deadline:
                break
                
            # Linearly decrease inertia weight
            w = w_start - (w_start - w_end) * (t / max(1, total_iters - 1))
            
            for i in range(N):
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
                
                # Clamp velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Clip to box [-5, 5]
                X[i] = np.clip(X[i], -5, 5)
                
                # Evaluate
                F_new = func(X[i])
                calls += 1
                
                # Update personal best
                if F_new < F_p_best[i]:
                    F_p_best[i] = F_new
                    P_best[i] = X[i].copy()
                    
                    # Update global best
                    if F_new < F_g_best:
                        F_g_best = F_new
                        G_best = X[i].copy()
                        
        return G_best, F_g_best
# EVOLVE-BLOCK-END
