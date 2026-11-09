"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: adaptive exploration + Nelder-Mead refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_remaining, deadline):
        """Run Nelder-Mead simplex method with a given budget."""
        n = self.dim
        # Initial simplex with adaptive step size based on current best
        simplex = [x0.copy()]
        # Create additional vertices with smaller steps for refinement
        for i in range(n):
            v = x0.copy()
            step = 0.1 if x0[i] < 0 else -0.1
            v[i] += step
            simplex.append(v)

        # Evaluate initial simplex
        fvals = np.array([func(np.clip(v, -5, 5)) for v in simplex])
        calls_used = n + 1

        # Track best
        best_idx = np.argmin(fvals)
        best_f = fvals[best_idx]
        best_x = simplex[best_idx].copy()

        # Nelder-Mead parameters tuned for Rosenbrock
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
            if spread < 1e-14:
                break

        return best_x, best_f, calls_used

    def _adaptive_exploration(self, func, x0, budget_remaining, rng, deadline):
        """Adaptive (1+1)-ES style exploration to find good basins."""
        n = self.dim
        x = x0.copy()
        fx = func(x)
        calls_used = 1
        best_x = x.copy()
        best_f = fx
        sigma = 1.5
        window = []
        window_size = 8

        while calls_used < budget_remaining:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y
                fx = fy
                window.append(1)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                window.append(0)

            if len(window) >= window_size:
                success_rate = np.mean(window)
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.5:
                    sigma *= 0.7
                window = window[-window_size // 2:] if window_size > 2 else []

            # Early stopping if sigma is very small
            if sigma < 1e-4:
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

        # Phase 1: Adaptive exploration with diverse starting points
        # Use a mix of structured and random starts
        init_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 1.5,
            np.ones(self.dim) * -1.5,
            np.ones(self.dim) * 3.0,
            np.ones(self.dim) * -3.0,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
        ]

        # Shuffle to diversify which starts get more budget
        idx = np.arange(len(init_points))
        rng.shuffle(idx)

        # Phase 1: Allocate ~40% of budget to adaptive exploration
        phase1_budget = int(total_budget * 0.4)
        exploration_results = []

        for start_idx in idx:
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break

            x0 = np.clip(init_points[start_idx], -5, 5)
            budget_for_exploration = min(
                phase1_budget - (calls_made - len(exploration_results)),
                total_budget - calls_made
            )
            if budget_for_exploration < 5:
                continue

            bx, bf, used = self._adaptive_exploration(func, x0, budget_for_exploration, rng, deadline)
            calls_made += used
            exploration_results.append((bf, bx.copy()))

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        if not exploration_results:
            # Fallback: just do a single Nelder-Mead from origin
            if calls_made < total_budget:
                bx, bf, used = self._nelder_mead(func, np.zeros(self.dim), total_budget - calls_made, deadline)
                calls_made += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()
            return best_x, best_f

        # Sort exploration results by function value
        exploration_results.sort(key=lambda x: x[0])

        # Phase 2: Nelder-Mead refinement on top candidates
        # Allocate remaining budget to top 2-3 candidates
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        n_refinements = min(len(exploration_results), max(2, len(exploration_results) // 4))
        budget_per_refinement = remaining // n_refinements

        for k in range(n_refinements):
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break

            _, x0 = exploration_results[k]
            budget_for_this = min(budget_per_refinement, total_budget - calls_made)

            if budget_for_this < 15:
                break

            bx, bf, used = self._nelder_mead(func, x0, budget_for_this, deadline)
            calls_made += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Final polish with very fine Nelder-Mead from best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 20:
                bx, bf, used = self._nelder_mead(func, best_x, remaining, deadline)
                calls_made += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
