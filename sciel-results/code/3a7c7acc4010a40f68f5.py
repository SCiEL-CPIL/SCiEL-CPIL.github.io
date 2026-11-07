"""Surrogate-assisted optimizer: RBF + EI for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import Rbf
import time


class Optimizer:
    """RBF surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling
        n_init = 50
        X = np.random.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        # Sort by fitness, keep best
        order = np.argsort(Y)
        X = X[order]
        Y = Y[order]

        best_x = X[0].copy()
        best_f = Y[0]

        # Track all evaluations
        calls_made = n_init

        # Phase 2: Surrogate-based optimization
        # Use RBF with inverse distance weights
        # We'll iteratively add points using EI

        # Parameters for EI
        xi = 0.01  # exploration-exploitation tradeoff
        n_ei_iters = 100  # max EI iterations before re-sampling
        max_retrain = 5  # retrain frequency

        # We'll do a hybrid: use EI for a while, then switch to local search
        n_surrogate_phase = 300

        for i in range(n_surrogate_phase):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Retrain RBF every max_retrain points
            if i % max_retrain == 0:
                try:
                    rbf = Rbf(X, Y, function='multiquadric')
                except Exception:
                    # Fallback: use nearest neighbor
                    break

            # Generate candidate points using EI
            n_candidates = 200
            candidates = np.random.uniform(-5, 5, size=(n_candidates, dim))
            # Also add points near current best
            for _ in range(50):
                candidates = np.vstack([candidates, best_x + rng.normal(0, 0.5, dim)])

            candidates = np.clip(candidates, -5, 5)

            # Evaluate surrogate
            try:
                Y_pred = rbf(candidates)
            except Exception:
                Y_pred = np.full(len(candidates), best_f)

            # Compute EI
            # EI = (best_f - Y_pred - xi) * Phi((best_f - Y_pred - xi)/sigma) + sigma * phi(...)
            # For RBF, we don't have uncertainty, so use a heuristic:
            # Points closer to best have lower "uncertainty"
            # Use distance-based uncertainty
            dists = np.linalg.norm(candidates - best_x, axis=1)
            max_dist = np.max(dists) + 1e-8
            sigma = 0.1 * best_f * (dists / max_dist)  # uncertainty increases with distance

            mu = Y_pred
            sigma = np.maximum(sigma, 1e-8)

            z = (best_f - mu - xi) / sigma
            from scipy.stats import norm
            EI = (best_f - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            EI = np.maximum(EI, 0)

            # Select best EI point
            best_idx = np.argmax(EI)
            x_new = candidates[best_idx]

            # Evaluate
            f_new = func(x_new)
            calls_made += 1

            # Update data
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, f_new])

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        # Phase 3: Local polish with (1+1)-ES around best point
        remaining = budget - calls_made
        if remaining > 0:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
