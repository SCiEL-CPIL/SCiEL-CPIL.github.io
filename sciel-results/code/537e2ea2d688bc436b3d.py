"""Optimized Multi-start Nelder-Mead for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Strategy: Multi-start Nelder-Mead
        # We divide the budget into chunks for different starts.
        # If a start yields a good result, we might spend more time refining it.
        # But for simplicity and robustness, we do fixed-size chunks.
        
        num_starts = 5
        calls_per_start = budget // num_starts

        for start_idx in range(num_starts):
            # Random start
            x0 = rng.uniform(-4, 4, size=dim)
            
            # Initial Simplex
            # Use a larger step to ensure we cover the landscape
            step = 4.0 
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0
                simplex[i + 1, i] += step
            
            # Evaluate initial simplex
            f_simplex = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls >= budget:
                    break
                x_i = np.clip(simplex[i], -5, 5)
                f_simplex[i] = func(x_i)
                calls += 1
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = x_i.copy()
            
            if calls >= budget:
                break

            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Run Nelder-Mead until budget for this start is exhausted or converged
            while calls < budget:
                # Sort
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Convergence check
                if np.max(f_simplex) - np.min(f_simplex) < 1e-6:
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha * (x_centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = x_centroid + gamma * (x_r - x_centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        x_c = x_centroid + rho * (x_r - x_centroid)
                    else:
                        x_c = x_centroid - rho * (x_worst - x_centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        # Need to check if we have enough budget for d evaluations
                        if calls + dim > budget:
                            break
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            x_shrink = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(x_shrink)
                            calls += 1
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = x_shrink.copy()
                        
                        if calls >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
