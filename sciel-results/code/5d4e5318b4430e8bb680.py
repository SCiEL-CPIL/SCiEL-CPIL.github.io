import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        t_start = time.time()
        max_time = 295.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        evals = 0

        # Restart CMA-ES parameters
        max_no_improve = 30  # Restart after 30 generations without improvement
        restart_sigma_factor = 1.5

        # We will run CMA-ES instances until budget or time runs out
        while evals < total_budget:
            if time.time() - t_start > max_time:
                break

            # CMA-ES parameters for this instance
            # Larger population for better exploration
            lam = 12
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w**2)

            # Initial mean and sigma
            # Start from the best point found so far, or 0 if none
            if best_x is None:
                m = np.zeros(n)
                sigma = 3.0  # Larger initial step size to cover [-5, 5] faster
            else:
                m = best_x.copy()
                sigma = 1.0  # Smaller step size for refinement after restart? Or keep adaptive?
                # Actually, on restart, we often want to explore a bit more broadly around the best point
                sigma = max(sigma, 0.5) 

            # Strategy parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

            # Evolution paths
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Covariance matrix
            C = np.eye(n)
            
            # Pre-compute Cholesky for C = I
            C_sqrt = np.linalg.cholesky(C)
            inv_C_sqrt = np.linalg.inv(C_sqrt)

            rng = np.random.default_rng(42 + evals) # Different seed for each restart? Or fixed?
            # Fixed seed is reproducible, but different restarts might benefit from different randomness.
            # However, the task implies deterministic evaluation or just a single run.
            # Let's use a fixed seed for reproducibility of the algorithm logic, 
            # but the restart points depend on the search history.
            
            local_best_x = m.copy()
            local_best_f = np.inf
            no_improve_count = 0
            gen = 0

            # Main Loop for this CMA-ES instance
            while evals + lam <= total_budget:
                if time.time() - t_start > max_time:
                    break

                gen += 1
                # Generate offspring
                X = np.zeros((lam, n))
                F = np.zeros(lam)

                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x = m + sigma * y
                    X[i] = x
                    F[i] = func(x)
                    evals += 1

                    if F[i] < local_best_f:
                        local_best_f = F[i]
                        local_best_x = x.copy()
                        if F[i] < best_f:
                            best_f = F[i]
                            best_x = x.copy()

                # Check for improvement
                if local_best_f < best_f:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]

                # Compute weighted mean of selected points
                y_weighted = np.zeros(n)
                for i in range(mu):
                    y_weighted += w[i] * (X[i] - m) / sigma

                # Update Step-size Control (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

                g = gen
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * g)) < (1.4 + 2/(n+1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update Covariance
                pc = (1 - c_c) * pc + c_c * y_weighted
                delta_sigma = c_sigma * (2 - c_sigma) * mueff

                C_new = (1 - c_1 - c_mu) * C
                C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)

                for i in range(mu):
                    z_i = (X[i] - m) / sigma
                    C_new += c_mu * w[i] * np.outer(z_i, z_i)

                # Ensure positive semi-definite
                C_new = (C_new + C_new.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                    C = C_new
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.linalg.cholesky(C)
                    inv_C_sqrt = np.linalg.inv(C_sqrt)

                # Update Mean
                m = m + sigma * y_weighted

                if sigma < 1e-10 or sigma > 1e10:
                    break

                # Check for stagnation
                if no_improve_count >= max_no_improve:
                    break

        return best_x, best_f
