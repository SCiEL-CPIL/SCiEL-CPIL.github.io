"""Optimized continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with 1/5 success rule and adaptive sigma.
    Simple, fast, and effective for black-box continuous optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Start near center for fast initial convergence
        x = np.zeros(dim)
        calls_made = 0

        def time_left():
            return (deadline - time.time()) > 0.5

        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            return x, fx

        x, fx = eval_point(x)
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        success_count = 0
        window_size = 10  # 1/5 rule window

        while calls_made < budget and time_left():
            # Generate step direction
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            y, fy = eval_point(y)

            if fy < fx:
                # Success
                x = y
                fx = fy
                success_count += 1

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure
                pass

            # 1/5 success rule: check every window_size steps
            if success_count % window_size == 0:
                # If exactly 1/5 of last window were successes, keep sigma
                # But we track cumulative in window, so reset counter
                # Actually, simpler: check rate over last window_size steps
                # Let's use a sliding window approach
                pass

            # Simpler 1/5 rule implementation:
            # Track successes in a fixed-size window
            # For efficiency, let's just do periodic checks
            if (calls_made - 1) % 10 == 0 and calls_made > 1:
                # Check success rate over last 10 steps
                # This is approximate but fast
                pass

            # Alternative: simple adaptive sigma
            # Use the fact that if we're making progress, increase sigma slightly
            # If not, decrease it

            # Let's use a simpler approach: track success rate over last 10 evaluations
            # We'll maintain a small history
            if not hasattr(self, '_history'):
                self._history = []
                self._last_x = x.copy()
                self._last_f = fx

            # Actually, let's restart with a cleaner implementation
            break

        # Restart with cleaner 1/5 rule
        # Reset state
        x = np.zeros(dim)
        x, fx = eval_point(x)
        best_x = x.copy()
        best_f = fx
        sigma = 1.0

        # Success history for 1/5 rule
        success_history = []
        window = 10

        while calls_made < budget and time_left():
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            y, fy = eval_point(y)

            if fy < fx:
                x = y
                fx = fy
                success_history.append(1)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_history.append(0)

            # Keep only last `window` entries
            if len(success_history) > window:
                success_history.pop(0)

            # 1/5 rule: if success rate > 0.2, increase sigma; if < 0.2, decrease
            if len(success_history) == window:
                rate = sum(success_history) / window
                if rate > 0.2:
                    sigma *= 1.05
                elif rate < 0.2:
                    sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Clean up
        if hasattr(self, '_history'):
            del self._history

        return best_x, best_f
# EVOLVE-BLOCK-END
