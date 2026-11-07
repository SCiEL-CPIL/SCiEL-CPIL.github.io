"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for the bent cigar function f12.
    
    The bent cigar has a very elongated valley: f(x) = sum(x_i^2) + 100*sum(x_i*x_{i+1})
    (approximately). The key challenge is the high condition number. CMA-ES handles
    this well through covariance matrix adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 300s limit, keep 20s margin

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))  # ~15 for dim=5
        # Step size
        sigma = 2.0
        # Initial mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, dim)
        
        # Evaluate initial point
        m = np.clip(m, -5, 5)
        fm = func(m)
        best_x = m.copy()
        best_f = fm

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        C_inv_sqrt = np.eye(dim)
        # Eigen decomposition for updating
        D = np.ones(dim)
        B = np.eye(dim)

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Adaptation rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Cholesky factor of C
        L = np.eye(dim)

        call_count = 1  # We already made 1 call
        it = 0

        while call_count < budget:
            # Check deadline
            if time.time() > deadline - 2:
                break

            # Sample lambda offspring
            y_list = []
            for i in range(lam):
                # Check deadline periodically
                if time.time() > deadline - 2:
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_i = np.clip(m + sigma * y, -5, 5)
                y_list.append(x_i)
                call_count += 1

            if len(y_list) < lam:
                # Pad if we ran out of time
                while len(y_list) < lam:
                    y_list.append(m.copy())

            # Evaluate offspring
            f_list = []
            for x_i in y_list:
                f_i = func(x_i)
                f_list.append(f_i)

            # Sort by fitness
            idx = np.argsort(f_list)
            y_list = [y_list[i] for i in idx]
            f_list = [f_list[i] for i in idx]

            # Update best
            if f_list[0] < best_f:
                best_f = f_list[0]
                best_x = y_list[0].copy()

            # Update mean
            new_m = m + sum(weights[i] * (y_list[i] - m) for i in range(mu))

            # Update covariance matrix
            # Compute delta
            delta = (new_m - m) / sigma

            # Update evolution path for covariance
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ delta
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) / \
                   np.sqrt(1 - (1 - c_sigma) ** (it + 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1)

            # Update evolution path for mean
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta

            # Update C
            delta_pc = (1 - c_c) * (c_c + 1) * np.outer(pc, pc)
            rank_mu = sum(weights[i] * np.outer(y_list[i] - m, y_list[i] - m) for i in range(mu))
            C = (1 + c_1 * (2 - c_1 - delta_pc[0, 0] / dim) + c_mu) * C - \
                c_1 * (np.outer(pc, pc) + delta_pc) - c_mu * rank_mu

            # Regularize C to maintain positive definiteness
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(dim)

            # Eigen decomposition
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
                B = B2
                C_inv_sqrt = B @ np.diag(1.0 / D2) @ B.T
            except:
                # Fallback if eigen decomposition fails
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                C_inv_sqrt = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            m = new_m
            m = np.clip(m, -5, 5)

            # If we're stuck, restart with larger sigma
            if it % 50 == 49 and sigma < 0.1:
                sigma = max(sigma, 1.0)

            it += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
