"""Rewritten optimizer for BBOB f13 (Sharp ridge) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive optimizer with deadline guard for f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Initial random sampling to find a decent starting region ---
        # Sample 20 points to get a good initial center
        n_init = min(20, budget // 10)
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0

        for _ in range(n_init - 1):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        calls = n_init

        # --- Phase 2: CMA-ES-like population-based search ---
        # Use a population-based evolution strategy with adaptive step size
        # Population size ~ 4 + floor(3 * log(dim))
        lam = max(4, 4 + int(3 * np.log(dim)))
        # Initial sigma based on best spread
        sigma = 1.5

        # CMA-ES parameters
        cm = np.eye(dim) * sigma ** 2  # covariance (diagonal initially)
        # For simplicity, use diagonal CMA-ES (safer for small budgets)
        diag = np.full(dim, sigma)

        # Success rate step control
        # Track successes over a window
        window = lam
        successes = 0
        step_size = sigma

        # Memory of best
        x_best = best_x.copy()
        f_best = best_f

        # Phase 2: Adaptive population search
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Generate population
            # Use diagonal step sizes per dimension
            candidates = []
            for i in range(lam):
                # Mix: some exploration, some exploitation
                if i == 0:
                    # Best point + small perturbation
                    y = np.clip(x_best + step_size * 0.3 * rng.standard_normal(dim), -5, 5)
                elif i < lam // 2:
                    # Medium step
                    y = np.clip(x_best + step_size * rng.standard_normal(dim), -5, 5)
                else:
                    # Larger step for exploration
                    y = np.clip(x_best + step_size * 2.0 * rng.standard_normal(dim), -5, 5)

                fy = func(y)
                calls += 1
                candidates.append((fy, y))

                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()

            # Sort candidates
            candidates.sort(key=lambda t: t[0])

            # Success rate step size control
            # Count how many improved over the best-so-far at start of generation
            # Use a simple rule: if top-half average is better than x_best, increase step
            top_half = candidates[:lam // 2]
            avg_top = np.mean([c[0] for c in top_half])

            # Compare with previous best
            if avg_top < f_best + 1e-10:
                step_size *= 1.3
            else:
                step_size *= 0.8

            # Clamp step size
            step_size = max(0.01, min(step_size, 5.0))

            # Also try to move x_best toward the best candidate
            # Weighted recombination
            weights = np.exp(-np.arange(lam) * 0.5) / np.sum(np.exp(-np.arange(lam) * 0.5))
            new_center = np.zeros(dim)
            for w, (_, y) in zip(weights, candidates):
                new_center += w * y
            # Blend with current best
            x_best = 0.5 * x_best + 0.5 * new_center

            # Update best if recombination is better
            # (we don't evaluate the recombined point to save budget, but we track it)

            # Adaptive: if we've made good progress, tighten
            if f_best < -50:
                step_size = max(step_size * 0.7, 0.01)

        # --- Phase 3: Local exploitation with small steps ---
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 2.0:
            # Fine-tune around the best point
            local_sigma = max(step_size * 0.1, 0.01)
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                y = np.clip(x_best + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return x_best, f_best
# EVOLVE-BLOCK-END
