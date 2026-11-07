"""Rewritten optimizer using Multistart Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multistart Nelder-Mead optimizer for BBOB f17 (Schaffers F7) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5, 5)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            x = clip(x)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        tol = 1e-10
        
        # Number of evaluations per NM run
        max_evals_per_run = 100
        
        # Initial delta for simplex
        current_delta = 2.0

        while calls < budget:
            # Determine start point for this run
            if calls == 0:
                x_start = np.zeros(n)
            else:
                # Strategy: Start near the best point found so far, with some noise
                if best_x is not None:
                    noise = rng.normal(0, current_delta, n)
                    x_start = clip(best_x + noise)
                else:
                    x_start = rng.uniform(-4, 4, n)
            
            # Create simplex
            simplex = [x_start]
            for i in range(n):
                v = x_start.copy()
                v[i] += current_delta
                simplex.append(v)
            
            # Evaluate simplex
            simplex_f = [evaluate(s) for s in simplex]
            if calls >= budget:
                break

            # Run Nelder-Mead
            run_evals = 0
            while run_evals < max_evals_per_run and calls < budget:
                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Convergence check
                if simplex_f[-1] - simplex_f[0] < tol:
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha * (x_centroid - x_worst)
                f_r = evaluate(x_r)
                run_evals += 1
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = x_centroid + gamma * (x_r - x_centroid)
                    f_e = evaluate(x_e)
                    run_evals += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        x_c = x_centroid + 0.5 * (x_r - x_centroid)
                    else:
                        x_c = x_centroid - 0.5 * (x_worst - x_centroid)
                    
                    f_c = evaluate(x_c)
                    run_evals += 1
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex_f[i] = evaluate(simplex[i])
                            run_evals += 1
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break
                
                if calls >= budget:
                    break

            # After a run, shrink the delta for next restart to explore locally
            current_delta *= 0.5
            if current_delta < 1e-4:
                current_delta = 1e-4

        return best_x, best_f
# EVOLVE-BLOCK-END
