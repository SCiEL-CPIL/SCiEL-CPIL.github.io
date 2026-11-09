"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_left():
            return time.time() < deadline

        # Multi-restart strategy: split budget into segments
        # Use larger segments for later restarts (exploit more)
        # First restart: broad exploration, later: fine-tuning near best

        # Define restart points
        # We'll do multiple restarts, each with a portion of the budget
        # Strategy: 6 restarts with decreasing step sizes
        n_restarts = 6
        # Budget allocation: first restart gets more for exploration
        # But for f21 (101 peaks), we need good coverage
        # Let's use roughly equal segments but with adaptive behavior
        budgets = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                b = remaining
            else:
                # Decreasing allocation: later restarts get less
                frac = 1.0 / (n_restarts - i)
                b = int(remaining * frac * 0.9)  # leave 10% for next
                b = max(b, 10)
            budgets.append(b)
            remaining -= b

        # Adjust to sum to budget
        diff = budget - sum(budgets)
        budgets[-1] += diff

        for restart_idx, seg_budget in enumerate(budgets):
            if not time_left() or calls_used >= budget:
                break

            # Initialize position for this restart
            if restart_idx == 0:
                # First restart: uniform random in box
                x = rng.uniform(-4.5, 4.5, n)
            else:
                # Subsequent restarts: perturb the best found so far
                # Use a random direction with some magnitude
                if best_x is not None:
                    # Mix: 30% chance to go near best, 70% to explore
                    if rng.random() < 0.3:
                        x = best_x + rng.normal(0, 0.5, n)
                    else:
                        # Random point but biased toward center of box
                        x = rng.uniform(-4.0, 4.0, n)
                        # Blend with best
                        alpha = 0.2
                        x = alpha * best_x + (1 - alpha) * x
                else:
                    x = rng.uniform(-4.5, 4.5, n)

            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma for this segment
            sigma = 1.5 if restart_idx == 0 else 0.8

            for _ in range(seg_budget - 1):
                if not time_left() or calls_used >= budget:
                    break

                # Generate candidate
                step = sigma * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success (1/5 rule approximation)
                    sigma *= 1.25
                    sigma = min(sigma, 3.0)
                else:
                    sigma *= 0.85
                    sigma = max(sigma, 0.01)

                # Occasionally re-center toward best if far away
                if restart_idx > 0 and rng.random() < 0.05:
                    if best_x is not None:
                        dist = np.linalg.norm(x - best_x)
                        if dist > 2.0:
                            # Move closer to best
                            x = 0.5 * x + 0.5 * best_x
                            x = np.clip(x, -5, 5)
                            fx = func(x)
                            calls_used += 1
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
