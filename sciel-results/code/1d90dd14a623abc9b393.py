"""Optimizer for BBOB-style f11 (Discus) using a two-phase radial-sweep + (1+1)-ES hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return float(func(np.clip(np.asarray(x, dtype=float), -5, 5)))

        def alive():
            return calls[0] < budget and time.time() < deadline

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float).copy()

        rng = np.random.default_rng(12345)

        # ---- Phase 0: Dense radial sweep to identify the needle axis ----
        # The Discus function has extreme anisotropy. We do a structured radial sweep
        # from the origin to quickly find the direction of the narrow valley.

        # First, sample the origin and a few basic directions
        origin = np.zeros(d)
        f_origin = wrapped(origin)
        update_best(origin, f_origin)

        # Radial sweep: sample points along many directions at several radii
        # Use a quasi-random set of directions for coverage
        n_dirs = 40
        n_radii = 5

        # Generate directions using a Fibonacci sphere / quasi-random approach
        directions = []
        for i in range(n_dirs):
            # Simple deterministic direction generation
            if i == 0:
                dir = np.ones(d) / np.sqrt(d)
            elif i <= d:
                dir = np.zeros(d)
                dir[i - 1] = 1.0
            elif i <= 2 * d:
                dir = np.zeros(d)
                dir[(i - d - 1) % d] = 1.0
                dir[(i - d - 1 + 1) % d] = 1.0
                dir /= np.sqrt(2)
            else:
                dir = rng.normal(0, 1, d)
                norm = np.linalg.norm(dir)
                if norm > 0:
                    dir /= norm
                else:
                    dir = np.ones(d) / np.sqrt(d)
            directions.append(dir)

        radii = [0.5, 1.0, 2.0, 3.0, 4.0]

        for radius in radii:
            if not alive():
                break
            for dir in directions:
                if not alive():
                    break
                pt = radius * dir
                pt = np.clip(pt, -5, 5)
                fp = wrapped(pt)
                update_best(pt, fp)

        # ---- Phase 1: Local refinement with a (1+1)-ES with adaptive step size ----
        # Now that we have a good initial point, use an aggressive (1+1)-ES
        # with high mutation rate to rapidly descend into the narrow valley.

        if best_x is None:
            best_x = np.zeros(d)

        x = best_x.copy()
        f = best_f
        sigma = 0.5  # initial step size

        # (1+1)-ES with 1/5 rule for step size adaptation
        success_count = 0
        total_count = 0
        max_steps = budget - calls[0] - 10

        for step in range(max_steps):
            if not alive():
                break

            # Generate mutant
            mutation = rng.normal(0, sigma, d)
            x_trial = x + mutation
            x_trial = np.clip(x_trial, -5, 5)

            f_trial = wrapped(x_trial)
            total_count += 1

            if f_trial < f:
                # Success
                x = x_trial
                f = f_trial
                update_best(x, f)
                success_count += 1
                # Adapt step size: increase on success
                sigma *= 1.2
                if sigma > 5.0:
                    sigma = 5.0
            else:
                # Failure: decrease step size
                sigma *= 0.8
                if sigma < 1e-6:
                    sigma = 1e-6

            # Periodically apply the 1/5 success rule for more robust adaptation
            if total_count % 10 == 0 and total_count >= 10:
                success_rate = success_count / total_count
                if success_rate < 0.2:
                    sigma *= 0.7
                elif success_rate > 0.8:
                    sigma *= 1.3
                success_count = 0
                total_count = 0

            # If we've converged very tightly, try a random restart with smaller radius
            if sigma < 1e-5:
                # Try a few random perturbations
                for _ in range(3):
                    if not alive():
                        break
                    pert = rng.normal(0, 0.1, d)
                    x_trial = x + pert
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped(x_trial)
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        update_best(x, f)
                        sigma = 0.1
                        break
                else:
                    # No improvement, break out of ES
                    break

        # ---- Phase 2: Final polish with a few more targeted evaluations ----
        if alive() and calls[0] < budget - 5:
            # Try a few small perturbations around the best point
            for scale in [0.01, 0.005, 0.001]:
                if not alive():
                    break
                for _ in range(3):
                    if not alive():
                        break
                    pert = rng.normal(0, scale, d)
                    x_trial = best_x + pert
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped(x_trial)
                    update_best(x_trial, f_trial)

        return best_x, best_f
# EVOLVE-BLOCK-END
