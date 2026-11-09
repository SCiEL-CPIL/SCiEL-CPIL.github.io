"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse lattice sampling + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(0)
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_fx, best_x
            if call_count >= total_budget:
                return best_fx
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x_clipped))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse lattice-based global sampling
        # Use stratified sampling to cover the space efficiently
        # For 5D, use a grid with ~5 points per dimension would be 3125 (too many)
        # Use ~150 points via stratified random sampling
        n_phase1 = 150
        phase1_budget = min(n_phase1, total_budget)

        # Generate stratified lattice points
        # Divide each dimension into bins and pick one random point per bin
        n_bins = 5  # 5 bins per dimension -> 5^5 = 3125 cells, pick 150
        lattice_points = []
        for i in range(phase1_budget):
            if time.time() > deadline:
                break
            # Stratified: divide [-5, 5] into 5 bins of width 2
            # Pick a random cell and a random point within it
            cell = rng.integers(0, n_bins, size=n)
            x = -5.0 + (cell + rng.random(n)) * (10.0 / n_bins)
            lattice_points.append(x)

        # Evaluate all lattice points
        for x in lattice_points:
            if time.time() > deadline or call_count >= total_budget:
                break
            safe_func(x)

        # Identify top 3 distinct basins from phase 1
        # Collect all evaluated points and their values
        # We need to track which points we've evaluated - let's restructure
        # Actually, let's track them properly

        # Reset and redo with proper tracking
        # Clear and restart phase 1 with tracking
        # For simplicity, let's just use the best few points from phase 1

        # Phase 2: Nelder-Mead refinement from top basins
        # We need to identify distinct basins. Since we don't have all points stored,
        # let's use a different approach: run Nelder-Mead from the current best
        # and a few perturbed versions

        remaining_budget = total_budget - call_count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_fx

        # Run Nelder-Mead from best point and 2 perturbed neighbors
        n_restarts_nm = 3
        budget_per_nm = max(1, remaining_budget // n_restarts_nm)

        # Starting points for Nelder-Mead
        start_points = []
        if best_x is not None:
            start_points.append(best_x.copy())
            # Add perturbed versions
            for _ in range(n_restarts_nm - 1):
                perturbed = best_x + rng.normal(0, 1.5, n)
                perturbed = np.clip(perturbed, -4.5, 4.5)
                start_points.append(perturbed)
        else:
            for _ in range(n_restarts_nm):
                start_points.append(rng.uniform(-4, 4, n))

        for sp in start_points:
            if time.time() > deadline or call_count >= total_budget:
                break
            self._nelder_mead(func, sp, budget_per_nm, n, deadline, safe_func)

        return best_x, best_fx

    def _nelder_mead(self, func, x0, budget, n, deadline, safe_func):
        """Nelder-Mead simplex method with time guard."""
        # Initialize simplex
        alpha = 1.0  # expansion
        gamma = 2.0  # reflection
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            dx = x0.copy()
            dx[i] += 0.1 if x0[i] < 4.9 else -0.1
            simplex.append(dx)

        # Evaluate initial simplex
        fx_values = []
        for i in range(n + 1):
            if time.time() > deadline:
                return
            fx_values.append(safe_func(simplex[i]))

        calls_used = n + 1
        budget_left = budget - calls_used

        # Nelder-Mead iterations
        while budget_left > 0 and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fx_values)
            simplex = [simplex[i].copy() for i in order]
            fx_values = [fx_values[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + gamma * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fxr = safe_func(xr)
            budget_left -= 1

            if fxr < fx_values[0]:
                # Expansion
                xe = centroid + alpha * (xr - centroid)
                xe = np.clip(xe, -5.0, 5.0)
                fxe = safe_func(xe)
                budget_left -= 1
                if fxe < fxr:
                    simplex[-1] = xe
                    fx_values[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx_values[-1] = fxr
            elif fxr < fx_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                fx_values[-1] = fxr
            else:
                # Contraction
                if fxr < fx_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fxc = safe_func(xc)
                budget_left -= 1

                if fxc <= min(fxr, fx_values[-1]):
                    simplex[-1] = xc
                    fx_values[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fx_values[i] = safe_func(simplex[i])
                        budget_left -= 1
                        if budget_left <= 0 or time.time() > deadline:
                            return
# EVOLVE-BLOCK-END
