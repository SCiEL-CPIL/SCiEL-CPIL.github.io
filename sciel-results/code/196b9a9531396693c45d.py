"""Rewritten optimizer using CMA-ES with restarts for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and integrated restart rule."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initial state
        x0 = np.zeros(dim)
        sigma0 = 2.0

        # CMA-ES parameters
        lambda_ = 4  # population size
        mu = lambda_ // 2  # number of recombination weights
        # Weights: standard CMA-ES weights
        weights = np.ones(mu)
        weights[-1] = 0.5 * weights[-1] if mu > 1 else weights[-1]
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Hessian correction
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Restart criteria
        max_iterations = 100  # max iterations per restart before checking stagnation
        stagnation_threshold = 50  # if no improvement in this many iterations, restart
        min_sigma = 1e-10
        max_sigma = 100.0

        def run_cma(x_center, sigma_start, remaining_budget):
            """Run CMA-ES from x_center with sigma_start for remaining_budget evaluations."""
            x = x_center.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            best_x = x.copy()
            best_f = func(best_x)
            eval_count = 1
            no_improve = 0
            prev_best = best_f

            while eval_count < remaining_budget:
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_):
                    # Sample from N(x, sigma^2 * C)
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    z = np.random.randn(dim)
                    y = x + sigma * B @ z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    eval_count += 1
                    if eval_count >= remaining_budget:
                        break

                if len(ys) < lambda_:
                    break

                ys = np.array(ys)
                fs = np.array(fs)

                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Mean update
                w = weights[:len(ys)]
                y_w = np.sum(w[:, None] * ys[:mu], axis=0)
                x_new = y_w

                # Success rate step size adaptation
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * np.linalg.norm(np.linalg.cholesky(C + 1e-10 * np.eye(dim))) + 1e-15) * (x_new - x)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (remaining_budget - eval_count + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, min_sigma, max_sigma)

                # Covariance matrix update
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * ((1 - c_c) ** 2 + hsig * c_c * (2 - c_c)) * np.outer(pc, pc)
                     + c_mu * np.sum(w[:, None, None] * np.outer(ys[:mu] - x, ys[:mu] - x) / sigma ** 2, axis=0))

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x = x_new

                # Restart check
                if no_improve > stagnation_threshold or sigma < min_sigma * 10:
                    break

            return best_x, best_f, eval_count

        # Main loop with restarts
        total_evals = 0
        best_x_overall = None
        best_f_overall = np.inf
        x_center = np.zeros(dim)
        sigma_start = 2.0
        restart_count = 0
        max_restarts = 5

        while total_evals < budget and restart_count <= max_restarts:
            remaining = budget - total_evals
            if remaining <= 0:
                break

            x_best, f_best, evals_used = run_cma(x_center, sigma_start, remaining)
            total_evals += evals_used

            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best.copy()

            # Check if we should restart
            if total_evals >= budget:
                break

            # If we made good progress, continue; otherwise restart
            if best_f_overall < -50:  # Good progress, continue
                x_center = best_x_overall.copy()
                sigma_start = 0.5
            else:
                # Restart from a different point
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Try a new random starting point
                rng = np.random.default_rng(restart_count)
                x_center = rng.uniform(-3, 3, dim)
                sigma_start = 1.5

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
