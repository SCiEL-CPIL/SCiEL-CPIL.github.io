"""Portfolio search for BBOB f7 (Step ellipsoidal) with plateau-aware (1+1)-ES and random restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Optimizer:
    """Portfolio of (1+1)-ES with proper 1/5 success rule and plateau-aware restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track global best
        best_x = rng.uniform(-5.0, 5.0, dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Limited random sampling to detect plateau vs. ellipsoid interior
        # For step ellipsoidal, most of the domain is plateau. We need to find the boundary.
        # Use fewer random samples but with strategic placement.
        n_random = 15
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of uniform and center-biased sampling
            if i < 5:
                x = rng.uniform(-5.0, 5.0, dim)
            elif i < 10:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-1.5, 1.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Detect if we're likely on the plateau
        # If all sampled values are very close, we're on the plateau
        sample_values = []
        # We already have best_f; check if we found any variation
        # For now, proceed to ES with plateau-aware behavior

        # Phase 2: (1+1)-ES with plateau-aware adaptation
        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # Start with larger sigma to escape plateau

        # 1/5 success rule: maintain a window of last 15 outcomes
        window_size = 15
        outcome_window = deque(maxlen=window_size)
        no_improve = 0
        max_no_improve = 25  # Lower patience for plateau
        restart_count = 0

        # Plateau detection: track if function values are all identical
        plateau_count = 0
        last_distinct_f = fx
        plateau_threshold = 8  # If 8+ consecutive non-improvements with same f, likely plateau

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            # Record outcome in window
            if fy < fx - 1e-10:
                outcome_window.append(1)
                # Improvement
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                plateau_count = 0
            else:
                outcome_window.append(0)
                no_improve += 1
                # Plateau detection: if fy is very close to fx, count as plateau
                if abs(fy - fx) < 1e-6:
                    plateau_count += 1
                else:
                    plateau_count = 0

            # Apply 1/5 success rule: adapt sigma based on mean of window
            if len(outcome_window) >= window_size:
                success_rate = sum(outcome_window) / len(outcome_window)
                if success_rate > 0.25:
                    sigma *= 1.3  # More aggressive increase
                elif success_rate < 0.15:
                    sigma *= 0.7  # More aggressive decrease
                # else: keep sigma unchanged

            # Clamp sigma
            sigma = min(max(sigma, 0.005), 8.0)

            # Plateau-aware restart: if we detect plateau, do a big jump
            if plateau_count >= plateau_threshold or no_improve >= max_no_improve:
                restart_count += 1

                # Adaptive restart strategy for step ellipsoidal:
                # On plateau, we need to find the ellipsoid boundary.
                # Strategy: try multiple scales of perturbation from best_x
                # and from origin (since minimizer is in [-4,4]^5)

                # Try a few different restart strategies
                found_improvement = False

                # Strategy 1: Large jump from current best
                for scale in [3.0, 2.0, 1.0]:
                    if calls >= budget or time.time() > deadline:
                        break
                    trial = best_x + scale * rng.standard_normal(dim)
                    trial = np.clip(trial, -5.0, 5.0)
                    f_trial = func(trial)
                    calls += 1
                    if f_trial < best_f - 1e-10:
                        best_f = f_trial
                        best_x = trial.copy()
                        x = trial.copy()
                        fx = f_trial
                        found_improvement = True
                        break
                    elif f_trial < fx - 1e-10:
                        x = trial.copy()
                        fx = f_trial

                # Strategy 2: If no improvement, try random restart
                if not found_improvement:
                    if rng.random() < 0.5:
                        x = rng.uniform(-5.0, 5.0, dim)
                    else:
                        x = rng.uniform(-2.0, 2.0, dim)  # Bias toward center where minimizer likely is
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Reset ES state
                sigma = 1.5
                no_improve = 0
                outcome_window.clear()
                plateau_count = 0

                # Decrease patience for future restarts
                max_no_improve = max(15, 25 - 3 * restart_count)
                plateau_threshold = max(5, 8 - restart_count)

        return best_x, best_f
# EVOLVE-BLOCK-END
