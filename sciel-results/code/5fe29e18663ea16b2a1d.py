"""Fast multi-start adaptive (1+1)-ES for BBOB Weierstrass f16.

Abandons surrogate modeling in favor of direct search with aggressive
restarts and two-stage polishing. This approach is faster, more robust,
and uses the full budget for exploration and exploitation.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with restarts and multi-stage polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)
        deadline = time.time() + 275.0
        n_calls = [0]

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        # ---- Phase 1: Diverse Initial Starts --------------------------------
        # Keep initial starts low to preserve budget for search.
        starts = [np.zeros(dim)]
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Add some structured starts
        for sign in [1, -1]:
            starts.append(np.full(dim, sign * 2.0))
            starts.append(np.full(dim, sign * 4.0))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            f = safe(s)
            if np.isfinite(f):
                record(s, f)

        # ---- Phase 2: Adaptive (1+1)-ES with Restarts -----------------------
        # Use 80% of remaining budget for the main search phase.
        main_limit = int(n_calls[0] + (budget - n_calls[0]) * 0.8)

        sigma = 1.5
        window = []
        win_size = 10
        stall_threshold = 30  # Lower threshold for quicker restarts
        max_restarts = 10
        restart_count = 0
        no_improve = 0

        while n_calls[0] < main_limit and time.time() < deadline:
            no_improve = 0
            while n_calls[0] < main_limit and time.time() < deadline and no_improve <= stall_threshold:
                cand = best_x + sigma * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_x = cand.copy()
                    best_f = f
                    window.append(1)
                    no_improve = 0
                else:
                    window.append(0)
                    no_improve += 1

                # Sigma adaptation
                if len(window) >= win_size:
                    sr = sum(window) / win_size
                    # Faster adaptation: 1.3 / 0.75
                    if sr > 0.35:
                        sigma *= 1.3
                    elif sr < 0.12:
                        sigma *= 0.75
                    window = window[-win_size // 2:]
                
                sigma = np.clip(sigma, 1e-9, 6.0)

            # Restart if stalled
            if n_calls[0] < main_limit and time.time() < deadline and no_improve > stall_threshold:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Alternate between random restarts and restarts near best
                if restart_count % 2 == 1:
                    start_x = rng.uniform(-4.0, 4.0, dim)
                else:
                    start_x = best_x + rng.uniform(-1.0, 1.0, dim)
                
                start_x = np.clip(start_x, -5.0, 5.0)
                f = safe(start_x)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = start_x.copy()
                
                sigma = 1.0
                window = []

        # ---- Phase 3: Two-Stage Polish --------------------------------------
        # Use remaining 20% of budget for fine-tuning.
        if n_calls[0] < budget and time.time() < deadline:
            polish_budget = budget - n_calls[0]
            # Stage 1: Medium polish
            sigma_p1 = max(1e-3, sigma * 0.1)
            steps_p1 = int(polish_budget * 0.5)
            for _ in range(steps_p1):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                cand = best_x + sigma_p1 * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = cand.copy()
                    sigma_p1 *= 0.95
                else:
                    sigma_p1 *= 0.95
                sigma_p1 = max(sigma_p1, 1e-6)

            # Stage 2: Fine polish
            sigma_p2 = max(1e-5, sigma_p1 * 0.1)
            for _ in range(budget - n_calls[0]):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                cand = best_x + sigma_p2 * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = cand.copy()
                    sigma_p2 *= 0.9
                else:
                    sigma_p2 *= 0.95
                sigma_p2 = max(sigma_p2, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
