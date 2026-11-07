import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0

        # CMA-ES parameters tuned for dim=5, budget=500
        # Larger population for better exploration early
        lam = 15  # Population size
        mu = lam // 2  # Number of parents
        # Log-weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Chi-square approximation
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        def run_cma_es(x0, sigma0, remaining_budget):
            nonlocal best_x, best_f, evals

            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Initial Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Track best
            local_best_x = m.copy()
            local_best_f = func(m)
            evals += 1

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Stagnation tracking
            no_improve_count = 0
            stagnation_threshold = 20  # Generations without improvement

            gen = 0
            while evals + lam <= remaining_budget and (time.time() - start_time) < deadline:
                gen += 1

                # Sample lambda candidates using Cholesky
                Z = np.random.randn(lam, dim)
                # x = m + sigma * Z @ L.T
                X = m + sigma * (Z @ L.T)

                # Evaluate
                F = np.array([func(x) for x in X])
                evals += lam

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]

                # Update best
                if F_sorted[0] < local_best_f:
                    local_best_f = F_sorted[0]
                    local_best_x = X_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Check for stagnation - restart with perturbation
                if no_improve_count > stagnation_threshold and gen > 5:
                    # Perturb the mean and restart
                    break

                # Update mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Compute C^{-1/2} v using Cholesky
                v = (m_new - m) / sigma
                # Solve L * y = v => y = L^{-1} v
                try:
                    y = np.linalg.solve(L, v)
                    v_inv_half = y
                except np.linalg.LinAlgError:
                    v_inv_half = v

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half

                # Update sigma
                sigma_norm = np.linalg.norm(p_sigma) / chi_lambda
                sigma = sigma * np.exp(c_sigma / chi_lambda * (sigma_norm - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # h_sigma
                h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
                h_sigma = 1.0 if sigma_norm < h_sigma_thresh else 0.0

                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

                # Update C
                Y = (X_sorted[:mu] - m) / sigma
                M2 = (Y.T * weights) @ Y
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
                C = (C + C.T) / 2

                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize and retry
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                        C = np.eye(dim)

                # Update mean
                m = m_new

            return local_best_x, local_best_f

        # Phase 1: Initial CMA-ES run
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Larger initial step for exploration
        run_cma_es(x0, sigma0, budget)

        # Phase 2: If time and budget remain, try local refinement from best
        if evals < budget and (time.time() - start_time) < deadline:
            # Use a smaller sigma for local refinement
            local_sigma = max(0.1, best_f * 0.01 if best_f > 0 else 0.1)
            x_start = best_x.copy()
            run_cma_es(x_start, local_sigma, budget)

        return best_x, best_f
