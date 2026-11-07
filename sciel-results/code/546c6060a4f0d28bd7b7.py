"""Direct optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with multi-restart for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_cma_es(x0, max_calls):
            nonlocal calls_made, best_f, best_x, deadline

            # CMA-ES Parameters
            N = max(4, dim)  # Population size
            mu = N // 2     # Number of parents
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialization
            mean = x0.copy()
            sigma = 1.0
            C = np.eye(dim)
            # Precompute B and B_inv for initial identity C
            B = np.eye(dim)
            B_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Chi_N constant
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Main Loop
            while calls_made < max_calls:
                if time.time() > deadline:
                    break

                # Check if we have enough budget for one full generation
                if calls_made + N > max_calls:
                    # If not enough for full generation, do (1+1)-ES style steps?
                    # Or just break. Breaking is safer to avoid partial generation issues in CMA-ES.
                    # However, we can use remaining budget for local (1+1)-ES polish.
                    break

                # 1. Generate Population
                population = []
                for i in range(N):
                    z = rng.standard_normal(dim)
                    y = B @ z
                    x_i = mean + sigma * y
                    x_i = np.clip(x_i, -5.0, 5.0)
                    population.append(x_i)

                # 2. Evaluate Population
                f_vals = [eval_func(x) for x in population]

                # 3. Sort by Fitness
                sorted_idx = np.argsort(f_vals)
                population = [population[i] for i in sorted_idx]
                f_vals = [f_vals[i] for i in sorted_idx]

                # 4. Update Mean
                old_mean = mean.copy()
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += w[i] * population[i]
                mean = new_mean

                # 5. Update p_sigma
                # y_mean = (new_mean - old_mean) / sigma
                y_mean = (mean - old_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ y_mean)

                # 6. Update h_sigma
                # Threshold
                thres = (1 + 2 / (dim + 1)) * chi_n
                # Normalized norm of p_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Decay factor
                decay = 1 - (1 - c_sigma) ** (2)  # Approximation for single step, usually (1-c_sigma)^(2 * (iter+1))
                # For simplicity and speed, we use a standard check
                # h_sigma = 1 if norm_p_sigma / sqrt(1 - (1-c_sigma)^(2*(iter+1))) < thres else 0
                # Since we don't track iter globally easily in this loop structure without passing it, 
                # and (1-c_sigma) is close to 1, the denominator is approx 1.
                h_sigma = 1 if norm_p_sigma < thres else 0

                # 7. Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

                # 8. Update Covariance Matrix C
                # Rank-1 update
                # C = (1 + c_1 * h_sigma * (2 - c_sigma) + c_mu) * C
                # C -= c_1 * (1 + h_sigma * (2 - c_sigma) - c_c) * np.outer(p_c, p_c)
                
                # Rank-mu update
                # C -= c_mu * sum( w_i * outer(y_i, y_i) )
                
                # Combined update
                # Calculate y_i for selected parents
                y_parents = np.zeros((dim, mu))
                for i in range(mu):
                    y_parents[:, i] = (population[i] - old_mean) / sigma

                # Update C
                C = (1 + c_1 * h_sigma * (2 - c_sigma) + c_mu) * C
                C -= c_1 * (1 + h_sigma * (2 - c_sigma) - c_c) * np.outer(p_c, p_c)
                
                # Subtract rank-mu part
                # sum w_i y_i y_i^T
                rank_mu_part = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu_part += w[i] * np.outer(y_parents[:, i], y_parents[:, i])
                C -= c_mu * rank_mu_part

                # Ensure positive definiteness
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # 9. Update Sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # 10. Update B and B_inv
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                B_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Multi-start CMA-ES
        # Start 1: Center
        start1 = np.zeros(dim)
        # Start 2: Random
        start2 = rng.uniform(-4, 4, dim)
        
        # Allocate budget
        # Use 60% for first start, 40% for second? 
        # Or 50/50.
        # Center is a good guess for BBOB.
        budget1 = int(budget * 0.6)
        budget2 = budget - budget1

        run_cma_es(start1, budget1)
        
        if calls_made < budget:
            run_cma_es(start2, budget2)

        # If we have remaining calls and time, do a (1+1)-ES polish
        if calls_made < budget and time.time() < deadline:
             # Simple (1+1)-ES on the current best
             x = best_x.copy()
             fx = best_f
             sigma = 0.5
             while calls_made < budget and time.time() < deadline:
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = func(y)
                 calls_made += 1
                 if fy < fx:
                     x, fx = y, fy
                     sigma = min(sigma * 1.1, 5.0)
                     if fx < best_f:
                         best_f = fx
                         best_x = x.copy()
                 else:
                     sigma = max(sigma * 0.9, 1e-5)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
