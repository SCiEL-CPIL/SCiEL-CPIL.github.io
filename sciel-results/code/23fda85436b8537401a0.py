"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma
        # Use 5 restarts with varying budgets to explore more of the search space
        n_restarts = 5
        per_restart = 80

        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Start point: center for first, random for others
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = eval_func(x)
            sigma = 2.0  # Larger initial sigma for Rastrigin

            # Success rate tracking for sigma adaptation
            window = 8
            recent = [False] * window
            idx = 0

            for _ in range(per_restart):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma based on success rate in recent window
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.45:
                        sigma *= 1.15
                    elif sr < 0.15:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.05, 4.0)

        # Phase 2: Local refinement using Nelder-Mead on best point
        if best_x is not None and calls < budget and time.time() < deadline:
            from scipy.optimize import minimize
            
            # Use Nelder-Mead for local refinement
            x_local = best_x.copy()
            f_local = best_f
            
            # Wrap func to track calls
            nm_calls = [0]
            def nm_func(x):
                nm_calls[0] += 1
                return func(np.clip(x, -5, 5))
            
            # Set options to limit function evaluations
            max_nm_evals = min(budget - calls, 50)
            try:
                result = minimize(nm_func, x_local, method='Nelder-Mead', 
                                 options={'maxiter': max_nm_evals, 'xatol': 0.01, 'fatol': 0.01})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
