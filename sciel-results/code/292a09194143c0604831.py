"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def halton_sequence(n, dim):
    """Generate n points from a Halton sequence in [0,1)^dim."""
    points = np.zeros((n, dim))
    for i in range(1, n + 1):
        for j in range(dim):
            base = j + 2  # primes: 2, 3, 5, ...
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                r += f * (i_val % base)
                i_val //= base
            points[i - 1, j] = r
    return points


class Optimizer:
    """Halton-sequence coarse-to-fine search with adaptive local ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Halton coarse search (first 60% of budget)
        n_halton = int(budget * 0.6)
        halton_points = halton_sequence(n_halton, dim)
        # Map [0,1)^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        # Also add a few random points for diversity
        n_random = min(10, budget - n_halton)
        random_points = rng.uniform(-5, 5, (n_random, dim))
        all_points = np.vstack([halton_points, random_points])

        # Local ES parameters
        sigma_local = 0.3
        local_success = 0
        local_attempts = 0
        local_x = None

        # We'll alternate: evaluate Halton points, and when we find a new best,
        # do a few local ES steps around it
        halton_idx = 0
        local_steps_since_halton = 0
        max_local_steps = 5  # do at most 5 local ES steps before returning to Halton

        while calls < budget:
            if time.time() > deadline:
                break

            # Decide whether to do a local ES step or a Halton point
            do_local = (
                local_x is not None
                and local_steps_since_halton < max_local_steps
                and local_steps_since_halton > 0
            )

            if do_local:
                # Local (1+1)-ES step around local_x
                if local_attempts >= 5:
                    rate = local_success / local_attempts
                    if rate > 0.3:
                        sigma_local *= 1.4
                    elif rate < 0.3:
                        sigma_local /= 1.4
                    sigma_local = max(sigma_local, 1e-6)
                    sigma_local = min(sigma_local, 2.0)
                    local_success = 0
                    local_attempts = 0

                z = rng.standard_normal(dim)
                x_trial = local_x + sigma_local * z
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                local_attempts += 1

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    local_x = x_trial.copy()
                    local_success += 1
                    local_steps_since_halton += 1
                else:
                    # No improvement; maybe move local center to best_x
                    local_x = best_x.copy()
                    local_steps_since_halton += 1

                if local_steps_since_halton >= max_local_steps:
                    local_steps_since_halton = 0
                    # Reset local sigma when returning to Halton
                    sigma_local = max(sigma_local * 0.5, 0.1)

            else:
                # Evaluate next Halton/random point
                if halton_idx < len(all_points):
                    x_trial = all_points[halton_idx].copy()
                    halton_idx += 1
                else:
                    # Exhausted pre-computed points; generate more random
                    x_trial = rng.uniform(-5, 5, dim)

                f_trial = func(x_trial)
                calls += 1

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    # Found new global best: switch to local refinement
                    local_x = x_trial.copy()
                    sigma_local = 0.3
                    local_steps_since_halton = 1
                    local_success = 0
                    local_attempts = 0

                # If we've been doing Halton for a while and have a good best,
                # occasionally do a local step
                if best_x is not None and calls > n_halton * 0.3 and halton_idx % 3 == 0:
                    local_x = best_x.copy()
                    local_steps_since_halton = 1

            # In the last 15% of budget, focus purely on local polish
            if calls > int(budget * 0.85) and best_x is not None:
                local_x = best_x.copy()
                sigma_local = max(sigma_local * 0.7, 0.01)
                local_steps_since_halton = 1
                max_local_steps = 50  # allow more local steps in polish phase

        return best_x, best_f
# EVOLVE-BLOCK-END
