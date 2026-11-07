"""Rewritten optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with CMA-like step-size adaptation for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_es(self, x0, fx0, budget_remaining, func, rng, start_time, deadline):
        """Run (1+1)-ES from x0 with given budget."""
        x = x0.copy()
        fx = fx0
        sigma = 0.5
        dim = self.dim

        # 1/5 success rule with larger window for smoother adaptation
        window_size = 15
        success_history = []
        calls_used = 0

        # Track best
        best_x = x.copy()
        best_f = fx

        # Adaptive step-size using cumulative success rate
        while calls_used < budget_remaining:
            if time.time() >= deadline:
                break

            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Track success
            success = fy < fx
            success_history.append(success)
            if len(success_history) > window_size:
                success_history.pop(0)

            # Adapt step size
            if len(success_history) == window_size:
                success_rate = sum(success_history) / window_size
                # Use a multiplicative update closer to CMA-ES style
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-10, 10.0)

            # Accept if improved
            if success:
                x = y
                fx = fy

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Multi-start strategy:
        # Phase 1: Coarse exploration with multiple random starts
        # Phase 2: For each promising start, run (1+1)-ES with remaining budget

        # Determine number of restarts and budget allocation
        # For 5D with 500 budget: try 8 starts with ~60 calls each for exploration,
        # then focus remaining budget on the best 2-3 starts

        n_starts = 8
        exploration_budget = 40  # calls per start for initial exploration
        total_exploration = n_starts * exploration_budget  # 320 calls

        # But we only have 500 total, so:
        # - 8 starts * 10 initial evals = 80 calls for coarse grid
        # - Then pick top 3 and run ES with remaining budget

        # Phase 1: Coarse exploration
        candidates = []
        calls_used = 0

        # Generate diverse starting points
        # Mix of random and structured points
        start_points = []

        # Random points
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, dim))

        # Structured: center and near-center
        start_points.append(np.zeros(dim))
        start_points.append(0.5 * rng.uniform(-1, 1, dim))
        start_points.append(1.0 * rng.uniform(-1, 1, dim))

        # Evaluate all start points
        for sp in start_points:
            if calls_used >= budget or not time_ok():
                break
            sp = np.clip(sp, -5, 5)
            fs = func(sp)
            calls_used += 1
            candidates.append((sp.copy(), fs))

        # Sort by function value
        candidates.sort(key=lambda c: c[1])

        # Phase 2: Run (1+1)-ES from the best few starts
        # Allocate remaining budget to top candidates
        remaining_budget = budget - calls_used
        top_k = min(3, len(candidates))

        # Allocate budget: give more to better candidates
        # Top 1 gets 50%, top 2 gets 30%, top 3 gets 20%
        if top_k == 1:
            budgets = [remaining_budget]
        elif top_k == 2:
            budgets = [int(0.6 * remaining_budget), int(0.4 * remaining_budget)]
        else:
            budgets = [int(0.5 * remaining_budget), int(0.3 * remaining_budget), int(0.2 * remaining_budget)]

        best_x = candidates[0][0].copy()
        best_f = candidates[0][1]

        for i in range(top_k):
            if calls_used >= budget or not time_ok():
                break

            x0, fx0 = candidates[i]
            b = min(budgets[i], budget - calls_used)
            if b <= 0:
                break

            bx, bf, used = self._run_es(x0, fx0, b, func, rng, start_time, deadline)
            calls_used += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
