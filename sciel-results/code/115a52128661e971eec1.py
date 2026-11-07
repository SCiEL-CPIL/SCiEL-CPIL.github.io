"""Multi-phase (1+1)-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase (1+1)-ES with adaptive step size and refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(0)
        n = self.dim

        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # Phase 1: Exploration with (1+1)-ES and adaptive sigma
        x = rng.uniform(-5, 5, n)
        fx = safe_eval(x)
        if fx is None:
            return best_x, best_f

        sigma = 2.0
        success_count = 0
        total_count = 0
        window_size = 20  # window for 1/5 success rule

        while calls_made < self.budget and time.time() < deadline - 5:
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = safe_eval(y)
            if fy is None:
                break

            total_count += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                success_count += 1
            else:
                sigma *= 0.87

            # 1/5 success rule: adjust sigma based on recent successes
            if total_count % window_size == 0:
                rate = success_count / window_size
                if rate > 0.2:
                    sigma *= 1.1  # more successes, increase step
                elif rate < 0.1:
                    sigma *= 0.9  # fewer successes, decrease step
                success_count = 0

            # Early termination if sigma becomes very small
            if sigma < 1e-10:
                break

        # Phase 2: Refinement with decreasing sigma
        if calls_made < self.budget and time.time() < deadline - 2 and best_x is not None:
            polish_x = best_x.copy()
            polish_f = best_f
            polish_sigma = 0.1
            polish_calls = 0
            max_polish_calls = min(self.budget - calls_made, 80)

            while polish_calls < max_polish_calls and time.time() < deadline - 1:
                step = polish_sigma * rng.standard_normal(n)
                candidate = np.clip(polish_x + step, -5, 5)
                candidate_f = safe_eval(candidate)
                if candidate_f is None:
                    break
                polish_calls += 1

                if candidate_f < polish_f:
                    polish_x = candidate
                    polish_f = candidate_f
                    polish_sigma *= 1.3
                else:
                    polish_sigma *= 0.75

                    # If no progress for a while, reduce sigma more aggressively
                    if polish_sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
