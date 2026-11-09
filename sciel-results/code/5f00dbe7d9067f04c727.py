"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-restart on Rosenbrock-style function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_remaining, rng, deadline):
        """Run Nelder-Mead simplex method with a given budget."""
        n = self.dim
        # Initial simplex
        simplex = [x0.copy()]
        # Create additional vertices by perturbing one coordinate at a time
        for i in range(n):
            v = x0.copy()
            step = 0.5 if x0[i] < 0 else -0.5
            v[i] += step
            simplex.append(v)

        # Evaluate initial simplex
        fvals = np.array([func(np.clip(v, -5, 5)) for v in simplex])
        calls_used = n + 1

        # Track best
        best_idx = np.argmin(fvals)
        best_f = fvals[best_idx]
        best_x = simplex[best_idx].copy()

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        # Main loop
        while calls_used < budget_remaining:
            if time.time() >= deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = fvals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
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
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= budget_remaining:
                            break

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

            # Check convergence: if simplex is very small, stop
            spread = np.max(fvals) - np.min(fvals)
            if spread < 1e-12:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        total_budget = self.budget
        calls_made = 0

        # Generate diverse starting points
        init_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 4.0,
            np.ones(self.dim) * -4.0,
            np.ones(self.dim) * 3.0,
            np.ones(self.dim) * -3.0,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
        ]

        # Shuffle
        idx = np.arange(len(init_points))
        rng.shuffle(idx)

        # Allocate budget: give more to promising starts
        # Use a greedy approach: evaluate each start, then allocate remaining budget
        # to the best-performing starts

        # First, evaluate all initial points
        init_results = []
        for i in range(len(init_points)):
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break
            x0 = np.clip(init_points[i], -5, 5)
            fx = func(x0)
            calls_made += 1
            init_results.append((fx, i, x0.copy()))

            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if not init_results:
            return best_x, best_f

        # Sort by function value
        init_results.sort(key=lambda x: x[0])

        # Allocate remaining budget to top starts
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Give budget to top 3-5 starts depending on how many we have
        n_starts = min(len(init_results), max(3, len(init_results) // 3))
        budget_per_start = remaining // n_starts

        for k in range(n_starts):
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break

            _, start_idx, x0 = init_results[k]
            budget_for_this = min(budget_per_start, total_budget - calls_made)

            if budget_for_this < 10:  # Not enough for a meaningful run
                break

            bx, bf, used = self._nelder_mead(func, x0, budget_for_this, rng, deadline)
            calls_made += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # If we still have budget, do a final local refinement from the best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 10:
                bx, bf, used = self._nelder_mead(func, best_x, remaining, rng, deadline)
                calls_made += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
