"""Multi-start adaptive (1+1)-ES with polish phase for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with polish phase for the BBOB Weierstrass (f16) task."""

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

        # Phase 2: adaptive (1+1)-ES with the 1/5 success rule
        # Step size adapts based on success rate over a window
        sigma = 2.0  # initial step size
        success_window = []
        window_size = 15

        # Main search phase: use ~70% of remaining budget
        main_budget_limit = int(n_calls[0] + (budget - n_calls[0]) * 0.7)
        
        while n_calls[0] < main_budget_limit and time.time() < deadline:
            # Generate a candidate
            candidate = best_x + sigma * rng.standard_normal(dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            fx = safe_func(candidate)

            if fx < best_f:
                # Success: move to candidate
                best_x = candidate.copy()
                best_f = fx
                success_window.append(1)
            else:
                success_window.append(0)

            # Adapt step size using 1/5 rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / window_size
                if success_rate > 0.3:
                    sigma *= 1.25
                elif success_rate < 0.2:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                success_window = success_window[-window_size // 2:]

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

        # Phase 3: High-precision polish
        # Use the remaining budget with a much smaller step size
        # and more aggressive sigma reduction for fine-tuning
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.1, 1e-4)
            polish_window = []
            polish_window_size = 5
            
            # Reset for polish phase
            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    # More aggressive reduction on success in polish phase
                    polish_sigma *= 0.7
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                # Adapt step size in polish phase
                if len(polish_window) >= polish_window_size:
                    success_rate = sum(polish_window) / polish_window_size
                    if success_rate > 0.4:
                        polish_sigma *= 1.3
                    elif success_rate < 0.15:
                        polish_sigma *= 0.75
                    polish_window = polish_window[-polish_window_size // 2:]

                # Keep polish_sigma in reasonable bounds
                polish_sigma = max(polish_sigma, 1e-8)
                polish_sigma = min(polish_sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
