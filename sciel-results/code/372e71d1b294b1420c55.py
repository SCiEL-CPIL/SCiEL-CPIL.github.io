"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        def safe_call(x):
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def nm_optimize(x0, remaining_budget):
            nonlocal best_x, best_f, calls_made
            if remaining_budget <= 0:
                return

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            fvals = np.zeros(dim + 1)

            for i in range(dim):
                step = 0.05 if x0[i] != 0 else 0.001
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step

            for i in range(dim + 1):
                fvals[i] = safe_call(simplex[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Main Nelder-Mead loop
            while calls_made < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check for convergence
                if np.std(fvals) < 1e-10 or np.max(np.abs(simplex - simplex[0])) < 1e-10:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_call(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_call(x_e)
                    calls_made += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_call(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

        # Multi-restart strategy
        n_restarts = 8
        budget_per_restart = max(50, budget // n_restarts)

        for r in range(n_restarts):
            if calls_made >= budget or time.time() >= deadline:
                break

            if r == 0:
                x0 = np.zeros(dim)
            else:
                # Use best found so far as base, add perturbation
                if best_x is not None:
                    scale = 0.5 * (1.0 - r / n_restarts)
                    x0 = best_x + scale * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            remaining = budget - calls_made
            nm_optimize(x0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
