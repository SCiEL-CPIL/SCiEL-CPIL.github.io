"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive ES for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(123)
        calls_used = [0]
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if calls_used[0] >= self.budget:
                raise TimeoutError()
            calls_used[0] += 1
            return func(x)

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Broad random sampling to find good basins
        n_initial = 30
        candidates = []
        for i in range(n_initial):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            try:
                f = safe_func(x)
            except TimeoutError:
                break
            candidates.append((f, x.copy()))
            update_best(x, f)

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive ES from top candidates with proportional budget allocation
        n_local = min(5, len(candidates))
        
        # Allocate remaining budget proportionally to quality (lower f gets more budget)
        remaining_budget = self.budget - calls_used[0]
        
        if remaining_budget <= 0:
            return best_x, best_f

        # Weight by inverse of (f - min_f + epsilon) to give more budget to better starts
        min_f = candidates[0][0]
        weights = []
        for i in range(n_local):
            f_i = candidates[i][0]
            # Smaller weight for worse starts
            w = 1.0 / (f_i - min_f + 1.0)
            weights.append(w)
        
        total_w = sum(weights)
        budgets = [int(remaining_budget * w / total_w) for w in weights]
        # Ensure at least some budget for each
        min_budget = max(5, remaining_budget // (n_local * 2))
        for i in range(n_local):
            budgets[i] = max(budgets[i], min_budget)
        
        # Adjust to fit within remaining budget
        total_alloc = sum(budgets)
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            budgets = [int(b * scale) for b in budgets]
            # Fix rounding
            diff = remaining_budget - sum(budgets)
            budgets[0] += diff

        for i in range(n_local):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break
            
            x = candidates[i][1].copy()
            f = candidates[i][0]
            sigma = 0.5  # initial step size
            local_budget = budgets[i]
            success_count = 0
            total_tries = 0
            success_rate = 0.5

            for _ in range(local_budget):
                if calls_used[0] >= self.budget or time.time() > deadline:
                    break
                
                # Generate multiple candidates per iteration for better exploration
                n_offspring = 3
                best_offspring = None
                best_offspring_f = float('inf')
                
                for _ in range(n_offspring):
                    if calls_used[0] >= self.budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except TimeoutError:
                        break
                    if fy < best_offspring_f:
                        best_offspring_f = fy
                        best_offspring = y.copy()
                
                if best_offspring is None:
                    break
                
                total_tries += 1
                if best_offspring_f < f:
                    x = best_offspring
                    f = best_offspring_f
                    update_best(x, f)
                    success_count += 1
                    sigma *= 1.2  # increase step size on success
                else:
                    sigma *= 0.8  # decrease step size on failure
                
                # Adaptive sigma based on success rate
                if total_tries >= 10:
                    success_rate = success_count / total_tries
                    if success_rate > 0.6:
                        sigma *= 1.1
                    elif success_rate < 0.3:
                        sigma *= 0.9
                    # Reset counters periodically
                    if total_tries >= 20:
                        success_count = 0
                        total_tries = 0

                # Early termination if converged
                if sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
