"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble with time-aware budget allocation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, fx0, max_iter, deadline, call_count):
        """Nelder-Mead simplex method with clipping to box."""
        dim = self.dim
        n = dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [fx0]
        call_count[0] += 1

        # Create simplex around x0
        for i in range(n):
            xi = x0.copy()
            step = 0.5 if x0[i] < 4.5 else -0.5
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            call_count[0] += 1
            simplex.append(xi)
            fvals.append(fi)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        min_step = 1e-10

        for it in range(max_iter):
            # Check time
            if time.time() > deadline:
                break
            if call_count[0] >= self.budget:
                break

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-12:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            call_count[0] += 1

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                call_count[0] += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                call_count[0] += 1

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        call_count[0] += 1

            # Check if simplex is degenerate
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < min_step:
                break

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], call_count[0]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        rng = np.random.default_rng(42)

        # Generate diverse starting points using Latin Hypercube Sampling
        n_starts = 5
        lhs_points = []
        for i in range(n_starts):
            # Each dimension gets a unique stratified sample
            u = np.linspace(0, 1, n_starts) + np.random.uniform(0, 1/n_starts, n_starts)
            perm = np.random.permutation(n_starts)
            u = u[perm]
            x = -5 + 10 * u
            lhs_points.append(x)

        # Add a few random points for diversity
        for _ in range(3):
            x = rng.uniform(-4, 4, dim)
            lhs_points.append(x)

        # Evaluate initial points and sort by fitness
        initial_evals = []
        for x in lhs_points:
            x = np.clip(x, -5, 5)
            fx = func(x)
            initial_evals.append((x, fx))

        # Sort by function value (best first)
        initial_evals.sort(key=lambda t: t[1])

        # Allocate budget among starts, giving more to better starts
        # Use a greedy approach: run Nelder-Mead on best starts first
        call_count = [0]
        best_x = initial_evals[0][0].copy()
        best_fx = initial_evals[0][1]
        call_count[0] = len(initial_evals)

        # Run Nelder-Mead on each start, allocating time proportionally
        # Give more iterations to starts that found better values
        n_starts_total = len(initial_evals)
        for i in range(n_starts_total):
            if time.time() > deadline:
                break
            if call_count[0] >= self.budget:
                break

            x0, fx0 = initial_evals[i]

            # Allocate iterations based on rank: better starts get more
            # First start gets ~40% of remaining, second ~25%, etc.
            remaining_budget = self.budget - call_count[0]
            if i == 0:
                max_iter = int(remaining_budget * 0.35)
            elif i == 1:
                max_iter = int(remaining_budget * 0.25)
            elif i == 2:
                max_iter = int(remaining_budget * 0.20)
            elif i == 3:
                max_iter = int(remaining_budget * 0.12)
            else:
                max_iter = int(remaining_budget * 0.08)

            max_iter = max(max_iter, 10)  # Minimum iterations

            x_best, fx_best, calls = self._nelder_mead(
                func, x0.copy(), fx0, max_iter, deadline, call_count
            )

            if fx_best < best_fx:
                best_x = x_best.copy()
                best_fx = fx_best

        return best_x, best_fx
# EVOLVE-BLOCK-END
