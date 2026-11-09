"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Halton Sequence Sampling + Nelder-Mead Refinement optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = budget
        evaluated_points = []
        evaluated_values = []

        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            evaluated_points.append(x_clipped.copy())
            evaluated_values.append(val)
            return val

        # Phase 1: Global Sampling using Halton Sequence (Quasi-Monte Carlo)
        # Allocate ~40% of budget to global exploration
        n_global = max(10, int(budget * 0.4))
        
        try:
            # Primes for Halton sequence
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            bases = primes[:dim]
            
            def radical_inverse(n, base):
                f = 1.0
                r = 0.0
                while n > 0:
                    f /= base
                    r += f * (n % base)
                    n //= base
                return r

            for i in range(1, n_global + 1):
                if calls_used >= max_calls:
                    break
                x = np.array([radical_inverse(i, b) for b in bases])
                # Map [0, 1]^dim to [-5, 5]^dim
                x = x * 10.0 - 5.0
                wrapped_func(x)
        except Exception:
            pass

        # Phase 2: Local Refinement using Nelder-Mead on top candidates
        n_refine = 5
        remaining_budget = max_calls - calls_used
        
        if remaining_budget > 0 and evaluated_points:
            # Sort indices by function value (ascending)
            indices = np.argsort(evaluated_values)
            # Select top K candidates
            k = min(n_refine, len(indices))
            top_indices = indices[:k]
            
            # Distribute remaining budget among top candidates
            calls_per_refine = max(1, remaining_budget // k)
            
            for idx in top_indices:
                if calls_used >= max_calls:
                    break
                start_x = evaluated_points[idx]
                try:
                    res = minimize(
                        wrapped_func,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': calls_per_refine, 'maxfev': calls_per_refine, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
