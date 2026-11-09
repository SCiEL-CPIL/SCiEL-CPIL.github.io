"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead with stretched simplex for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        d = self.dim
        budget = self.budget

        # Wrap func to track calls
        call_count = [0]

        def tracked_func(x):
            x = np.asarray(x, dtype=float)
            if x.shape != (d,):
                x = x.reshape(d)
            result = func(x)
            call_count[0] += 1
            return result

        best_x = None
        best_f = np.inf

        # Generate initial diverse points using Sobol sequence
        from scipy.stats import qmc
        n_sobol = min(64, budget - 20)
        sobol = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol) * 10.0 - 5.0

        # Evaluate Sobol points to find good starting locations
        candidates = []
        for i in range(n_sobol):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            x = np.clip(sobol_points[i], -5, 5)
            f = tracked_func(x)
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(d)
            f = tracked_func(x)
            best_x = x.copy()
            best_f = f

        # Sort candidates by function value (best first)
        candidates.sort(key=lambda t: t[0])

        # For each of the top candidates, run Nelder-Mead with a stretched simplex
        # aligned with the ridge direction (anisotropic initial simplex)
        n_starts = min(3, len(candidates))
        for idx in range(n_starts):
            if call_count[0] >= budget or time.time() > deadline:
                break

            start_x = candidates[idx][1]
            start_f = candidates[idx][0]

            # Create anisotropic initial simplex:
            # - Base point at start_x
            # - Stretch along one axis (ridge direction) by factor 5
            # - Other axes use standard small perturbations
            simplex = [start_x.copy()]
            simplex_f = [start_f]

            # Standard simplex size
            base_step = 0.5
            stretch_factor = 5.0

            for j in range(d):
                x_new = start_x.copy()
                if j == 0:
                    # Stretch first axis to align with potential ridge
                    x_new[j] += base_step * stretch_factor
                else:
                    x_new[j] += base_step
                x_new = np.clip(x_new, -5, 5)
                f_new = tracked_func(x_new)
                simplex.append(x_new.copy())
                simplex_f.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

                if call_count[0] >= budget or time.time() > deadline:
                    break

            if call_count[0] >= budget or time.time() > deadline:
                continue

            # Run Nelder-Mead from this stretched simplex
            try:
                res = minimize(
                    tracked_func,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'maxiter': 200,
                        'maxfev': max(10, budget - call_count[0]),
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
