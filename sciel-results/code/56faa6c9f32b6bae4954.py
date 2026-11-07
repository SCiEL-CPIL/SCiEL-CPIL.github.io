"""Optimizer for BBOB-style f17 (Schaffers F7) using Nelder-Mead with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts for low-dimensional optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(func, x0, sigma, max_iter, calls_ref):
            """
            Nelder-Mead simplex method.
            Returns (best_x, best_f, calls_made).
            """
            d = len(x0)
            n_calls = calls_ref[0]

            # Initialize simplex: x0 plus d vertices at distance sigma
            simplex = np.zeros((d + 1, d))
            simplex[0] = x0
            for i in range(d):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma

            # Evaluate initial simplex
            simplex_f = np.zeros(d + 1)
            for i in range(d + 1):
                if n_calls >= budget:
                    return simplex[0], simplex_f[0], n_calls
                if time.time() > deadline - 2.0:
                    return simplex[0], simplex_f[0], n_calls
                simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                n_calls += 1
                calls_ref[0] = n_calls

            # Track best in simplex
            best_idx = np.argmin(simplex_f)
            best_x_simplex = simplex[best_idx].copy()
            best_f_simplex = simplex_f[best_idx]

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            for iteration in range(max_iter):
                if n_calls >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                # Check convergence: if simplex is small enough
                centroid = np.mean(simplex[:-1], axis=0)
                spread = np.max(np.abs(simplex - centroid), axis=0)
                if np.max(spread) < 1e-8:
                    break

                # Reflection
                x_reflect = centroid - alpha * (simplex[-1] - centroid)
                x_reflect = np.clip(x_reflect, -5, 5)
                if n_calls >= budget:
                    break
                f_reflect = func(x_reflect)
                n_calls += 1
                calls_ref[0] = n_calls

                if f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid - gamma * (simplex[-1] - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    if n_calls >= budget:
                        break
                    f_expand = func(x_expand)
                    n_calls += 1
                    calls_ref[0] = n_calls

                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    # Simple reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < simplex_f[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (simplex[-1] - centroid)

                    x_contract = np.clip(x_contract, -5, 5)
                    if n_calls >= budget:
                        break
                    f_contract = func(x_contract)
                    n_calls += 1
                    calls_ref[0] = n_calls

                    if f_contract < min(f_reflect, simplex_f[-1]):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if n_calls >= budget:
                                break
                            simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                            n_calls += 1
                            calls_ref[0] = n_calls

                # Update best
                best_idx = np.argmin(simplex_f)
                if simplex_f[best_idx] < best_f_simplex:
                    best_f_simplex = simplex_f[best_idx]
                    best_x_simplex = simplex[best_idx].copy()

            return best_x_simplex, best_f_simplex, n_calls

        # Multi-start Nelder-Mead with different initial points and step sizes
        starts = [
            (np.zeros(dim), 1.0),
            (np.array([2.0] * dim), 1.5),
            (np.array([-2.0] * dim), 1.5),
            (rng.uniform(-3, 3, size=dim), 2.0),
        ]

        for x0, sigma_init in starts:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Estimate remaining budget for this start
            remaining = budget - calls_used
            # Allocate roughly equal budget to each start, but at least 20 calls
            max_iter_for_this = max(20, remaining // (len(starts) - starts.index((x0, sigma_init))))

            x_best, f_best, calls_made = nelder_mead(
                func, x0, sigma_init, max_iter_for_this, [calls_used]
            )
            calls_used = calls_made
            update_best(x_best, f_best)

        # Final polish with (1+1)-ES on the best point found
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_budget = min(50, budget - calls_used)

            for _ in range(polish_budget):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8

            update_best(x, fx)

        if best_x is None:
            # Fallback: just evaluate center
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
