"""Surrogate-assisted optimizer for BBOB f6: GP with Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-based surrogate optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s wall-clock limit

        dim = self.dim
        total_budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Initial sampling for surrogate training
        n_initial = 50
        X_samples = []
        f_samples = []

        for _ in range(n_initial):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)
            X_samples.append(x)
            f_samples.append(fx)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        X = np.array(X_samples)
        f = np.array(f_samples)

        # Phase 2: Surrogate-based optimization with Expected Improvement
        # Use a simple GP with RBF kernel
        # For efficiency, we'll use a simplified approach:
        # - Maintain a growing set of samples
        # - Use a simple distance-based surrogate (inverse distance weighting)
        # - Or use a lightweight GP approximation

        # Actually, let's use a simpler but effective approach:
        # Radial Basis Function surrogate with iterative refinement

        n_ei_iterations = total_budget - n_calls  # remaining budget
        ei_candidates = 100  # number of candidates to evaluate EI for each step

        for _ in range(n_ei_iterations):
            if n_calls >= total_budget or time.time() > deadline:
                break

            # Generate candidate points
            # Mix of: random points, points near best, and EI-optimized points
            candidates = []

            # 1. Random candidates
            for _ in range(30):
                candidates.append(rng.uniform(-5, 5, dim))

            # 2. Points near current best
            for _ in range(20):
                candidates.append(np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5))

            # 3. EI-optimized candidates using a simple surrogate
            # Use inverse distance weighting as a fast surrogate
            if len(X_samples) >= 2:
                for _ in range(50):
                    # Sample from a distribution that favors unexplored regions
                    # but also areas near good solutions
                    if rng.random() < 0.5:
                        # Near best
                        cand = np.clip(best_x + rng.standard_normal(dim) * 1.0, -5, 5)
                    else:
                        # Random
                        cand = rng.uniform(-5, 5, dim)
                    candidates.append(cand)

            candidates = np.array(candidates)

            # Compute surrogate predictions using IDW
            # For each candidate, predict f and uncertainty
            best_ei = -float('inf')
            best_cand = None

            # Compute distances from candidates to all samples
            # This can be expensive, so we'll use a subset if needed
            n_ref = min(len(X_samples), 50)
            idx_ref = rng.choice(len(X_samples), n_ref, replace=False) if len(X_samples) > n_ref else np.arange(len(X_samples))
            X_ref = X_samples[idx_ref]
            f_ref = f_samples[idx_ref]

            for cand in candidates:
                # Compute distances
                d = np.linalg.norm(X_ref - cand, axis=1)
                # IDW prediction
                eps = 1e-8
                w = 1.0 / (d + eps) ** 2
                w_norm = w / w.sum()
                f_pred = np.dot(w_norm, f_ref)
                
                # Uncertainty estimate: variance of weighted samples
                f_var = np.dot(w_norm, (f_ref - f_pred) ** 2)
                f_std = np.sqrt(f_var + 1e-10)

                # Expected Improvement
                if f_std < 1e-10:
                    ei = 0.0
                else:
                    z = (f_pred - best_f) / f_std
                    ei = (best_f - f_pred) * norm.cdf(z) + f_std * norm.pdf(z)

                if ei > best_ei:
                    best_ei = ei
                    best_cand = cand

            if best_cand is None:
                # Fallback: random point
                best_cand = rng.uniform(-5, 5, dim)

            best_cand = np.clip(best_cand, -5, 5)
            fx = func(best_cand)
            record(best_cand, fx)
            X_samples.append(best_cand)
            f_samples.append(fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
