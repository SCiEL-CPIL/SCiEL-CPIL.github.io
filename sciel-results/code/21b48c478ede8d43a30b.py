"""Rewritten optimizer: Diagonal natural-gradient (Rank-1 + Rank-mu) continuous optimizer."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Diagonal step-size adaptation via Rank-1 + Rank-mu natural gradient."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize
        x = np.zeros(d)
        fx = func(x)
        sigma = np.ones(d) * 2.0  # per-dimension step sizes (log-scale for stability)
        log_sigma = np.log(sigma)

        # Population parameters
        lam = 8
        mu = 4
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = (w.sum()) ** 2 / np.sum(w ** 2)

        # Learning rates
        c_sigma = 1.0 / (2 * d)
        c_adapt = 1.0 / (d + 3)

        # State
        p_sigma = 0.0
        h_sigma = np.sqrt(1 - np.exp(-2 / budget))

        calls_used = 1

        for gen in range(budget - 1):
            # Generate offspring
            offspring = np.empty((lam, d))
            f_offspring = np.empty(lam)
            for i in range(lam):
                z = rng.standard_normal(d)
                y = np.clip(x + sigma * z, -5, 5)
                f_offspring[i] = func(y)
                offspring[i] = y
                calls_used += 1
                if calls_used >= budget:
                    break

            # Sort by fitness
            idx = np.argsort(f_offspring)
            n_top = min(mu, len(idx))
            top_idx = idx[:n_top]
            top_offspring = offspring[top_idx]
            top_f = f_offspring[top_idx]

            # Weighted mean of top mu
            w_use = w[:n_top]
            w_sum = w_use.sum()
            mu_mean = np.sum(w_use[:, None] * top_offspring, axis=0) / w_sum

            # Update x
            x_old = x.copy()
            x = x_old + (mu_mean - x_old)  # Move to weighted mean

            # Evaluate new x? No, we don't have budget for that. Use best offspring instead.
            # Actually, let's use the best offspring as the new x if it's better
            best_i = top_idx[0]
            best_y = offspring[best_i]
            best_f = f_offspring[best_i]
            if best_f < fx:
                x, fx = best_y, best_f

            # Adapt log_sigma
            # Rank-1: based on the step from x_old to x_new
            step = x - x_old
            step_norm = np.linalg.norm(step, ord=2)

            # Rank-mu: diagonal covariance of top mu
            if n_top > 1:
                centered = top_offspring - mu_mean
                diag_var = np.sum(w_use[:, None] ** 2 * centered ** 2, axis=0) / w_sum
                diag_var = np.maximum(diag_var, 1e-12)
                target_log_sigma = 0.5 * np.log(diag_var)
                log_sigma = (1 - c_adapt) * log_sigma + c_adapt * target_log_sigma
            else:
                # Rank-1 only
                if step_norm > 1e-10:
                    # If improvement, increase sigma; else decrease
                    if best_f < fx:
                        log_sigma += c_sigma * 0.5
                    else:
                        log_sigma -= c_sigma * 0.5

            # Ensure sigma stays in reasonable range
            log_sigma = np.clip(log_sigma, np.log(0.01), np.log(10.0))
            sigma = np.exp(log_sigma)

        return x, fx
# EVOLVE-BLOCK-END
