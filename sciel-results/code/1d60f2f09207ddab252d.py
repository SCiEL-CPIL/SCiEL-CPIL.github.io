"""Deterministic grid-based global search followed by multi-start local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1 (75% budget): deterministic Sobol-like sequence sampling to map the landscape.
      Phase 2 (25% budget): Multi-start Nelder-Mead local refinement from top-k best points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, count, dim):
        """Generate a Sobol-like low-discrepancy sequence in [0, 1)^dim using scipy."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=False)
        points = sampler.random(count)
        return points

    def _coordinate_descent(self, func, x0, max_calls, tol=1e-8):
        """Simple coordinate descent refinement."""
        x = x0.copy()
        calls_used = 0
        step = 0.5
        dim = len(x)
        
        for iteration in range(max_calls):
            improved = False
            for d in range(dim):
                for direction in [-1, 1]:
                    if calls_used >= max_calls:
                        break
                    x_trial = x.copy()
                    x_trial[d] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_used += 1
                    f_current = func(x)
                    calls_used += 1
                    if f_trial < f_current - tol:
                        x = x_trial
                        improved = True
                        break
                if calls_used >= max_calls:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-10:
                    break
            if calls_used >= max_calls:
                break
        
        f_final = func(x)
        calls_used += 1
        return x, f_final, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Global search with Sobol sequence (75% of budget)
        n_global = int(budget * 0.75)
        n_global = max(n_global, 50)
        n_local = budget - n_global

        # Generate Sobol points in [0, 1)^dim
        try:
            sobol_pts = self._sobol_sequence(n_global, dim)
        except Exception:
            # Fallback to Halton
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]
            halton_pts = []
            for i in range(1, n_global + 1):
                point = []
                for d, p in enumerate(base):
                    f = 1.0
                    r = 0.0
                    n = i
                    while n > 0:
                        f /= p
                        r += f * (n % p)
                        n //= p
                    point.append(r)
                halton_pts.append(point)
            sobol_pts = np.array(halton_pts)

        # Map to [-5, 5]^dim
        sobol_pts = sobol_pts * 10.0 - 5.0

        # Evaluate all global points and track top-k
        k = 5  # Number of local refinement starts
        evals = []  # list of (f, x)
        
        for i in range(n_global):
            x = sobol_pts[i]
            fx = func(x)
            evals.append((fx, x.copy()))
        
        # Sort by function value and keep top-k
        evals.sort(key=lambda t: t[0])
        top_k = evals[:k]
        best_x = top_k[0][1].copy()
        best_f = top_k[0][0]

        # Phase 2: Multi-start local refinement
        if n_local > 0:
            calls_made = [0]
            max_calls_total = n_local
            calls_per_start = max(10, n_local // k)

            for start_idx, (f_start, x_start) in enumerate(top_k):
                if calls_made[0] >= max_calls_total:
                    break
                
                remaining = max_calls_total - calls_made[0]
                calls_this_start = min(calls_per_start, remaining)
                
                if calls_this_start < 5:
                    break

                def wrapped_func(x, _calls=calls_made, _remaining=calls_this_start):
                    if _calls[0] >= max_calls_total:
                        return np.inf
                    _calls[0] += 1
                    return func(np.clip(x, -5, 5))

                # Nelder-Mead
                options = {
                    'maxiter': calls_this_start,
                    'maxfev': calls_this_start,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                }

                try:
                    result = minimize(
                        wrapped_func,
                        x_start,
                        method='Nelder-Mead',
                        options=options
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

                # Coordinate descent polish if we have budget left
                remaining_after_nm = max_calls_total - calls_made[0]
                if remaining_after_nm >= 10:
                    cd_calls = min(20, remaining_after_nm)
                    try:
                        x_cd, f_cd, cd_used = self._coordinate_descent(
                            wrapped_func, best_x, cd_calls
                        )
                        if f_cd < best_f:
                            best_f = f_cd
                            best_x = x_cd.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
