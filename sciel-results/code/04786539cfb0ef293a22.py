"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine 1D coordinate search + Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        calls = 0
        max_calls = self.budget

        def call(x):
            nonlocal calls
            if calls >= max_calls:
                return float('inf')
            calls += 1
            return func(x)

        dim = self.dim

        # --- Phase 1: Coarse 1D grid search along each axis ---
        # For Rastrigin f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum is at x_i = 0 for all i (before shift/rotation)
        # But we have hidden shift, rotation, offset. So we need to find the basin.
        # Strategy: for each dimension, scan a coarse grid to find approximate minimum.

        # Start with a reasonable initial point
        x = np.zeros(dim)
        fx = call(x)
        best_x = x.copy()
        best_fx = fx

        # Coarse grid search along each axis
        # Rastrigin has period 2 in each dimension (cos(2*pi*x))
        # Scan from -5 to 5 with step ~0.5 first
        coarse_step = 0.5
        for d in range(dim):
            for val in np.arange(-5, 5.01, coarse_step):
                x[d] = val
                f = call(x)
                if f < best_fx:
                    best_fx = f
                    best_x = x.copy()

        # Fine grid search around the best found
        fine_step = 0.1
        for d in range(dim):
            center = best_x[d]
            for val in np.arange(center - 1.0, center + 1.01, fine_step):
                val = np.clip(val, -5, 5)
                x = best_x.copy()
                x[d] = val
                f = call(x)
                if f < best_fx:
                    best_fx = f
                    best_x = x.copy()

        # Even finer
        if time.time() < deadline and calls < max_calls:
            finer_step = 0.02
            for d in range(dim):
                center = best_x[d]
                for val in np.arange(center - 0.5, center + 0.51, finer_step):
                    val = np.clip(val, -5, 5)
                    x = best_x.copy()
                    x[d] = val
                    f = call(x)
                    if f < best_fx:
                        best_fx = f
                        best_x = x.copy()

        # --- Phase 2: Nelder-Mead polish ---
        if time.time() < deadline and calls < max_calls:
            x = best_x.copy()
            fx = best_fx
            # Initialize simplex
            alpha = 0.1
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                xi[i] += alpha if xi[i] + alpha <= 5 else -alpha
                simplex.append(xi)

            # Evaluate simplex
            simplex_f = [call(simplex[0])]
            for i in range(1, dim + 1):
                simplex_f.append(call(simplex[i]))

            # Nelder-Mead parameters
            alpha_n = 1.0  # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma_nm = 0.5  # shrink

            max_nm_iter = min(200, max_calls - calls)
            for _ in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break

                # Sort by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha_n * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call(x_e)
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
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call(x_c)
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = call(simplex[i])

                # Track best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_fx:
                    best_fx = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
