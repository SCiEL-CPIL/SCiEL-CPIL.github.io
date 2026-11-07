"""Multi-start adaptive (1+1)-ES with restarts and polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with restarts and polish phase for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0  # safety margin below the 330s kill

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: cheap multi-start to find a good basin early
        # Deterministic diverse starts: center, axis-aligned, and random
        starts = [np.zeros(dim)]
        for scale in [1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        for _ in range(20):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: adaptive (1+1)-ES with the 1/5 success rule and restarts on stagnation.
        # Restarts (from inspiration 2) let us escape Weierstrass local basins; the 1/5-rule
        # (from the current program) keeps step size well-tuned inside a basin.
        sigma = 2.0
        success_window = []
        window_size = 15
        no_improve = 0
        restart_threshold = 45
        max_restarts = 12
        restarts = 0
        restart_improved = False

        while n_calls[0] < budget and time.time() < deadline:
            candidate = best_x + sigma * rng.standard_normal(dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            fx = safe_func(candidate)

            if fx < best_f:
                best_x = candidate.copy()
                best_f = fx
                success_window.append(1)
                no_improve = 0
            else:
                success_window.append(0)
                no_improve += 1

            # Adapt step size using the 1/5 rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / window_size
                if success_rate > 0.3:
                    sigma *= 1.25
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_window = success_window[-window_size // 2:]

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Restart on stagnation: re-seed from a fresh random point to escape the basin
            if no_improve > restart_threshold and restarts < max_restarts:
                restarts += 1
                start_x = rng.uniform(-4.5, 4.5, dim)
                fx2 = safe_func(start_x)
                if fx2 < best_f:
                    best_f = fx2
                    best_x = start_x.copy()
                    restart_improved = True
                else:
                    # Keep exploring around the global best but with a fresh step size
                    best_x = best_x.copy()
                sigma = rng.uniform(1.0, 3.0)
                success_window = []
                no_improve = 0

        # Phase 3: high-precision polish of the global best.
        # Only spend the remaining budget here; the 1/5-rule already left us in a tight basin,
        # so a small, aggressively-reducing sigma refines the minimum for a better late n_i.
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.1, 1e-3)
            polish_window = []
            polish_window_size = 5

            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.7
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    success_rate = sum(polish_window) / polish_window_size
                    if success_rate > 0.4:
                        polish_sigma *= 1.3
                    elif success_rate < 0.15:
                        polish_sigma *= 0.75
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-8)
                polish_sigma = min(polish_sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
