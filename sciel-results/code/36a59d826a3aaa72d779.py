"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            try:
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                return float(val)
            except Exception:
                return None

        def nelder_mead(func, x0, max_calls, deadline):
            """Run Nelder-Mead from x0, returning (best_x, best_f, calls_used)."""
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            f_vals = np.zeros(n + 1)

            # First vertex
            f0 = safe_call(simplex[0])
            if f0 is None:
                return x0, float('inf'), 1
            f_vals[0] = f0

            # Build other vertices by perturbing each dimension
            step = 0.5
            for i in range(1, n + 1):
                simplex[i] = simplex[0].copy()
                if i <= n:
                    simplex[i, i - 1] += step
                fi = safe_call(simplex[i])
                if fi is None:
                    fi = f0 + 1.0
                    simplex[i] = simplex[0].copy()
                f_vals[i] = fi

            calls_used = n + 1
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Sort helper
            def sort_simplex():
                nonlocal simplex, f_vals
                idx = np.argsort(f_vals)
                simplex = simplex[idx]
                f_vals = f_vals[idx]

            sort_simplex()

            while calls_used < max_calls and time.time() < deadline:
                # Check convergence
                if np.max(f_vals) - np.min(f_vals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)
                calls_used += 1
                if fr is None:
                    break
                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    calls_used += 1
                    if fe is None:
                        fe = fr
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_call(xc)
                    calls_used += 1
                    if fc is None:
                        break
                    if fc <= min(fr, f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = safe_call(simplex[i])
                            calls_used += 1
                            if fi is None:
                                fi = f_vals[i]
                            f_vals[i] = fi
                        # Re-sort
                        sort_simplex()

                sort_simplex()

                # Track best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()

            return simplex[0].copy(), f_vals[0], calls_used

        # Multi-start strategy: try several starting points
        n_starts = 3
        per_budget = budget // n_starts

        for s in range(n_starts):
            if calls_made >= budget or time.time() >= deadline:
                break

            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            remaining = budget - calls_made
            if remaining <= 0:
                break

            x_best, f_best, _ = nelder_mead(func, x0, remaining, deadline)

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Final polish with a small (1+1)-ES around the best point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls_made):
                if time.time() >= deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
