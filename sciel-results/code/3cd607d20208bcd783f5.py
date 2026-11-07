"""Rewritten program: Particle Swarm Optimization with Mutation for BBOB f3."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization with Mutation continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # PSO Parameters
        M = 12  # Swarm size, 500 // 12 = 41 iterations
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        p_mut = 0.1  # Probability of random restart/mutation
        
        # Bounds
        lb = -5.0
        ub = 5.0
        range_ = ub - lb
        v_max = 0.1 * range_  # 10% of range per step
        
        # Initialize swarm
        x = rng.uniform(lb, ub, size=(M, n))
        v = rng.uniform(-v_max, v_max, size=(M, n))
        
        # Evaluate initial swarm
        fitness = np.array([func(x[i]) for i in range(M)])
        
        # Personal bests
        p_best = x.copy()
        p_best_fit = fitness.copy()
        
        # Global best
        g_best_idx = np.argmin(fitness)
        g_best = x[g_best_idx].copy()
        g_best_fit = fitness[g_best_idx]
        
        calls_made = M
        
        total_iters = budget // M
        
        for it in range(total_iters):
            # Update inertia weight
            w = w_start - (w_start - w_end) * (it / max(1, total_iters - 1))
            
            for i in range(M):
                # Update velocity
                r1 = rng.random(n)
                r2 = rng.random(n)
                
                v[i] = w * v[i] + c1 * r1 * (p_best[i] - x[i]) + c2 * r2 * (g_best - x[i])
                
                # Clamp velocity
                v[i] = np.clip(v[i], -v_max, v_max)
                
                # Update position
                x[i] = x[i] + v[i]
                
                # Mutation: Random restart
                if rng.random() < p_mut:
                    x[i] = rng.uniform(lb, ub, size=n)
                    v[i] = 0.0
                else:
                    x[i] = np.clip(x[i], lb, ub)
                
                # Evaluate
                f_new = func(x[i])
                calls_made += 1
                
                # Update personal best
                if f_new < p_best_fit[i]:
                    p_best_fit[i] = f_new
                    p_best[i] = x[i].copy()
                    
                    # Update global best
                    if f_new < g_best_fit:
                        g_best_fit = f_new
                        g_best = x[i].copy()
                        
            # If we have exceeded budget, break
            if calls_made >= budget:
                break
                
        return g_best, g_best_fit
# EVOLVE-BLOCK-END
