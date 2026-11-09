"""Rewritten program using Nelder-Mead with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts for f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best value found so far
        best_x = None
        best_f = np.inf

        # We'll use multiple restarts if budget allows
        # Each restart uses a portion of the budget
        # Strategy: try to find good solutions with Nelder-Mead, restart if stuck

        # Initial simplex creation
        def make_simplex(center, scale):
            """Create an initial simplex around center."""
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(1, dim + 1):
                simplex[i] = center.copy()
                simplex[i, i - 1] += scale
            return simplex

        def nelder_mead(func, x0, budget_left, scale=0.5):
            """Run Nelder-Mead optimization starting from x0 with given budget."""
            nonlocal best_x, best_f

            dim = self.dim
            simplex = make_simplex(x0, scale)
            fvals = np.array([func(np.clip(s, -5, 5)) for s in simplex])

            # Update global best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = simplex[idx_best].copy()

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            calls_used = dim + 1  # initial simplex evaluations
            iterations = 0
            max_iter = budget_left - dim - 1

            while calls_used < budget_left and iterations < max_iter:
                iterations += 1
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check for convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = func(np.clip(simplex[i], -5, 5))
                            calls_used += 1
                        if calls_used >= budget_left:
                            break

                # Update global best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < best_f:
                    best_f = fvals[idx_best]
                    best_x = simplex[idx_best].copy()

            return best_x, best_f, calls_used

        # Main search strategy:
        # 1. Start with a few random initial points to get a sense of the landscape
        # 2. Run Nelder-Mead from the best point
        # 3. If budget remains, do local refinement or restart from perturbed best

        # Phase 1: Random sampling to find a good starting point
        n_random = min(10, budget // 10)
        random_points = []
        for i in range(n_random):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            random_points.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        random_points.sort(key=lambda t: t[0])
        calls_used = n_random

        # Phase 2: Nelder-Mead from the best random point
        if calls_used < budget:
            x0 = random_points[0][1]
            scale = 1.0
            best_x, best_f, nm_calls = nelder_mead(func, x0, budget - calls_used, scale)
            calls_used += nm_calls

        # Phase 3: If budget remains, try refinement with smaller scale
        if calls_used < budget:
            # Try a few more Nelder-Mead runs from perturbed best point
            remaining = budget - calls_used
            n_refine = min(3, remaining // 10)
            for i in range(n_refine):
                if remaining <= dim + 1:
                    break
                # Perturb best point
                perturbation = 0.1 * (1 + i * 0.5)
                x0 = best_x + perturbation * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
                scale = 0.3 * (1 + i * 0.5)
                best_x, best_f, nm_calls = nelder_mead(func, x0, remaining, scale)
                calls_used += nm_calls
                remaining = budget - calls_used

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
