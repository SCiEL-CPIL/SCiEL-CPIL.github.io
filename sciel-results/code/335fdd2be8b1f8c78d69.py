"""Optimizer for BBOB f23 (Katsuura) using per-dimension CMA-ES on separable landscape."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Per-dimension CMA-ES with LHS initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        # Phase 1: Latin Hypercube sampling for diverse initial points
        n_lhs = min(40, budget // 12)
        lhs_points = []
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            pts = (perm + rng.uniform(size=n_lhs)) / n_lhs * 10.0 - 5.0
            lhs_points.append(pts)
        lhs_points = np.array(lhs_points).T  # shape (n_lhs, dim)

        best_x = None
        best_f = np.inf

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            best_x = x
            best_f = fx

        # Phase 2: Per-dimension CMA-ES (exploiting separability)
        # Each dimension is optimized independently with its own CMA-ES
        # Budget is shared: total calls must stay within budget

        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget per dimension
        calls_per_dim = max(1, remaining // dim)
        x_best = best_x.copy()

        for d in range(dim):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # CMA-ES on dimension d, holding other dims fixed
            # Use a small CMA-ES: mean, sigma, covariance (scalar in 1D)
            mean = x_best[d]
            sigma = 1.0
            lam = 4  # population size (small for budget efficiency)
            max_iters = calls_per_dim // lam + 1

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate lambda samples for this dimension
                samples = np.array([mean + sigma * rng.standard_normal() for _ in range(lam)])
                samples = np.clip(samples, -5, 5)

                # Evaluate each sample (changing only dimension d)
                vals = []
                for s in samples:
                    if time.time() > deadline:
                        break
                    if calls_made >= budget:
                        break
                    x_trial = x_best.copy()
                    x_trial[d] = s
                    f_trial = func(x_trial)
                    calls_made += 1
                    vals.append(f_trial)

                if len(vals) < lam:
                    # Pad with current best if we ran out of budget
                    while len(vals) < lam:
                        vals.append(best_f)

                vals = np.array(vals)
                order = np.argsort(vals)
                mu = lam // 2  # select top half

                # Update mean (weighted average of best mu)
                weights = np.exp(-np.arange(mu) / (mu / 2.0))
                weights /= weights.sum()
                new_mean = np.average(samples[order[:mu]], weights=weights)

                # Update sigma (1D: variance of selected)
                selected = samples[order[:mu]]
                new_sigma = np.std(selected) * 1.5  # slight boost
                new_sigma = np.clip(new_sigma, 1e-6, 5.0)

                # Accept if improvement
                best_sample_val = vals[order[0]]
                if best_sample_val < best_f:
                    x_best[d] = samples[order[0]]
                    best_f = best_sample_val

                mean = new_mean
                sigma = new_sigma

            # After CMA-ES on this dimension, update best_x
            x_best[d] = mean  # Use the final mean as the refined position
            # Evaluate to update best
            if calls_made < budget and time.time() < deadline:
                f_check = func(x_best)
                calls_made += 1
                if f_check < best_f:
                    best_f = f_check

        # Final: ensure we return the best found
        return x_best, best_f
# EVOLVE-BLOCK-END
