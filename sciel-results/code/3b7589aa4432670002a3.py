import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 295.0  # Safety margin before 300s limit

        # CMA-ES parameters
        # For n=5, lam=9 is standard
        lam = int(4 + 3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2

        # Weights for the top mu individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)

        # Mean and step size
        m = np.zeros(n)
        # Initial sigma: The box is [-5, 5]. A sphere of radius 2 covers most of the interesting space initially.
        sigma = 2.0

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

        rng = np.random.default_rng(42)

        evals = 0
        best_x = None
        best_f = np.inf

        # Initial evaluation
        x_init = m.copy()
        f_init = func(x_init)
        evals += 1
        best_x = x_init.copy()
        best_f = f_init

        # Main Loop
        while evals + lam <= budget:
            if time.time() - t_start > max_time:
                break

            # Generate offspring
            X = np.zeros((lam, n))
            F = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform to normal with covariance C
                y = C_sqrt @ z
                x = m + sigma * y
                X[i] = x
                F[i] = func(x)
                evals += 1

                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]

            # Compute weighted mean of selected points (in normalized space)
            # z_i = (x_i - m) / sigma
            # m_new = m + sigma * sum(w_i * z_i)
            y_weighted = np.zeros(n)
            for i in range(mu):
                y_weighted += w[i] * (X[i] - m) / sigma

            # Update Step-size Control (CSA)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * (C^-1/2 * y_weighted)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            # h_sigma gate: check if ps is within expected bounds
            # The term 1 - (1-c_sigma)^(2*g) approximates the variance of the path
            g = evals // lam + 1
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * g)) < (1.4 + 2/(n+1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            else:
                # If h_sigma is 0, we don't update sigma, but we also don't add the (1-delta_sigma)C term?
                # Actually, the standard update for C uses delta_sigma regardless, but the h_sigma gate is for sigma update.
                # In some implementations, if h_sigma=0, the covariance update is slightly different, but standard CMA-ES just skips sigma update.
                pass

            # Update Covariance
            pc = (1 - c_c) * pc + c_c * y_weighted

            # Standard CMA-ES covariance update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - delta_sigma) * C) + c_mu * sum w_i * (z_i)(z_i)^T
            delta_sigma = c_sigma * (2 - c_sigma) * mueff

            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)

            for i in range(mu):
                z_i = (X[i] - m) / sigma
                C_new += c_mu * w[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            C_new = (C_new + C_new.T) / 2
            try:
                # Add small regularization to ensure positive definiteness
                C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                inv_C_sqrt = np.linalg.inv(C_sqrt)
                C = C_new
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset covariance to identity or damp it
                # Resetting to identity is a robust fallback
                C = np.eye(n)
                C_sqrt = np.linalg.cholesky(C)
                inv_C_sqrt = np.linalg.inv(C_sqrt)

            # Update Mean
            m = m + sigma * y_weighted

            if sigma < 1e-10 or sigma > 1e10:
                break

        return best_x, best_f
