"""Rewritten optimizer using Nelder-Mead simplex with adaptive reset and multi-start diversification."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free trust-region / simplex-based optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Wrapper to count calls
        def counted_func(x):
            self._call_count += 1
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Quick random exploration to find promising regions
        n_random = min(30, budget // 4)
        candidates = []
        for _ in range(n_random):
            x0 = rng.uniform(-5, 5, dim)
            f0 = counted_func(x0)
            candidates.append((f0, x0.copy()))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        if self._call_count >= budget:
            return best_x, best_f

        # Adaptive Nelder-Mead with restarts
        def nelder_mead_adaptive(x_start, f_start, max_calls, initial_step=1.0):
            nonlocal self
            dim = self.dim

            # Initialize simplex with adaptive step size
            simplex_x = [x_start.copy()]
            simplex_f = [f_start]

            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_step if x_start[i] < 0 else -initial_step
                xi = np.clip(xi, -5, 5)
                fi = counted_func(xi)
                simplex_x.append(xi.copy())
                simplex_f.append(fi)

            best_f_so_far = min(simplex_f)
            best_x_so_far = simplex_x[int(np.argmin(simplex_f))].copy()
            no_improve = 0
            patience = 40
            step_size = initial_step

            while self._call_count < max_calls and no_improve < patience:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex_x = [simplex_x[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Convergence check
                f_spread = simplex_f[-1] - simplex_f[0]
                x_spread = np.max(np.abs(np.array(simplex_x) - simplex_x[0]))
                if f_spread < 1e-12 and x_spread < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex_x[:-1], axis=0)

                # Reflection
                gamma = 1.0
                xr = centroid + gamma * (centroid - simplex_x[-1])
                xr = np.clip(xr, -5, 5)
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    rho = 2.0
                    xe = centroid + rho * (centroid - simplex_x[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = counted_func(xe)
                    if fe < fr:
                        simplex_x[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex_x[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex_x[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    sigma = 0.5
                    if fr < simplex_f[-1]:
                        xc = centroid + sigma * (xr - centroid)
                    else:
                        xc = centroid - sigma * (centroid - xr)
                    xc = np.clip(xc, -5, 5)
                    fc = counted_func(xc)
                    if fc <= min(fr, simplex_f[-1]):
                        simplex_x[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex_x[i] = simplex_x[0] + 0.5 * (simplex_x[i] - simplex_x[0])
                            simplex_x[i] = np.clip(simplex_x[i], -5, 5)
                            simplex_f[i] = counted_func(simplex_x[i])

                # Update best
                cur_best_f = min(simplex_f)
                if cur_best_f < best_f_so_far - 1e-12:
                    best_f_so_far = cur_best_f
                    best_x_so_far = simplex_x[int(np.argmin(simplex_f))].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Adaptive step: shrink simplex if stuck
                if no_improve > patience // 2:
                    for i in range(1, dim + 1):
                        simplex_x[i] = simplex_x[0] + 0.5 * (simplex_x[i] - simplex_x[0])
                        simplex_x[i] = np.clip(simplex_x[i], -5, 5)
                        simplex_f[i] = counted_func(simplex_x[i])

            return best_x_so_far, best_f_so_far

        # Phase 2: Local optimization from top candidates
        n_local = min(3, len(candidates))
        for i in range(n_local):
            if self._call_count >= budget:
                break
            x_start = candidates[i][1].copy()
            f_start = candidates[i][0]
            x_local, f_local = nelder_mead_adaptive(x_start, f_start, budget)
            if f_local < best_f:
                best_f = f_local
                best_x = x_local

        # Phase 3: Final fine-tuning with scipy if budget remains
        if self._call_count < budget and best_x is not None:
            remaining = budget - self._call_count
            if remaining > 5:
                try:
                    result = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': remaining,
                            'xatol': 1e-9,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
