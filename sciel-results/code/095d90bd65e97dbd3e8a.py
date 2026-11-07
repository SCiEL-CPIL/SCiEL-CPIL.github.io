"""Grid-based coarse-to-fine optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Grid-based coarse-to-fine deterministic scan:
    1. Coarse grid over [-5,5]^dim to identify promising regions
    2. Progressive refinement: narrow to best region, increase resolution
    3. Local (1+1)-ES polish around the best grid point
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        def grid_scan(lo, hi, n_per_dim, budget_left):
            """Evaluate centers of an n_per_dim^dim grid over [lo, hi]^dim."""
            nonlocal calls_made, best_x, best_f
            if budget_left <= 0:
                return
            # Generate grid points (centers of cells)
            coords = np.linspace(lo, hi, n_per_dim + 2)[1:-1]  # cell centers
            # Iterate over all grid combinations
            indices = np.ndindex(*([n_per_dim] * self.dim))
            for idx in indices:
                if check_deadline():
                    break
                x = np.array([coords[i] for i in idx])
                evaluate(x)

        def local_polish(x0, f0, budget_left, sigma_init=0.5):
            """(1+1)-ES local refinement around x0."""
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            sigma = sigma_init
            success_streak = 0
            fail_streak = 0

            while budget_left > 0 and not check_deadline():
                y = x + sigma * rng.standard_normal(self.dim)
                y, fy = evaluate(y)
                budget_left -= 1
                if fy < f0:
                    x = y
                    f0 = fy
                    success_streak += 1
                    fail_streak = 0
                    # Expand slightly on success
                    if success_streak % 5 == 0:
                        sigma *= 1.1
                else:
                    fail_streak += 1
                    success_streak = 0
                    # Shrink on failure
                    if fail_streak % 3 == 0:
                        sigma *= 0.8
                sigma = np.clip(sigma, 0.01, 3.0)

        # Phase 1: Coarse grid scan over full domain
        # Allocate ~40% of budget for coarse exploration
        coarse_budget = max_calls * 2 // 5
        # For dim=5, a 3^5=243 grid is feasible within coarse_budget
        # Use 3 points per dimension for coarse scan
        n_coarse = 3
        grid_scan(-5.0, 5.0, n_coarse, coarse_budget)

        if best_x is None:
            # Fallback: evaluate origin
            evaluate(np.zeros(self.dim))
            return best_x, best_f

        # Phase 2: Medium-resolution refinement around the best region
        remaining = max_calls - calls_made
        if remaining > 20:
            # Narrow to a box around the best point
            half_width = 3.0  # search within [-5+3, 5-3] = [-2, 2] around best
            lo = np.maximum(-5.0, best_x - half_width)
            hi = np.minimum(5.0, best_x + half_width)
            # Use 4 points per dimension: 4^5 = 1024 points
            # But we may not have that many calls, so adapt
            n_medium = 4
            max_points = n_medium ** self.dim
            if max_points <= remaining * 2 // 3:
                grid_scan(lo[0], hi[0], n_medium, remaining * 2 // 3)
            else:
                # Fall back to fewer points per dim
                n_medium = 3
                grid_scan(lo[0], hi[0], n_medium, remaining * 2 // 3)

        # Phase 3: Fine-grained local polish using (1+1)-ES
        remaining = max_calls - calls_made
        if remaining > 10 and not check_deadline():
            local_polish(best_x, best_f, remaining, sigma_init=0.3)

        # Phase 4: Final ultra-fine polish with very small steps
        remaining = max_calls - calls_made
        if remaining > 5 and not check_deadline():
            local_polish(best_x, best_f, remaining, sigma_init=0.05)

        return best_x, best_f

# EVOLVE-BLOCK-END
