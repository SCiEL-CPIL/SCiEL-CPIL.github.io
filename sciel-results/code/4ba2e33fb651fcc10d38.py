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
        deadline = start_time + 285  # Safety margin before 300s limit

        # 1. Initial Exploration Phase
        # Sample a few random points to find a better starting location than the origin.
        # This is crucial for functions with hidden shifts.
        n_init = 10
        if budget <= n_init:
            # Fallback for extremely small budgets (though budget is 500)
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        X_init = np.random.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        
        best_idx = np.argmin(F_init)
        m = X_init[best_idx].copy()
        best_f = F_init[best_idx]
        best_x = X_init[best_idx].copy()
        evals = n_init

        # 2. CMA-ES Phase
        # Tuned parameters for dim=5
        # Standard heuristic: lambda = 4 + 3*ln(d). For d=5, ln(5)~1.6, 4+4.8=8.8 -> 9.
        # Let's try lambda=11 for robustness in 5D.
        lam = 11
        mu = lam // 2  # 5
        
        # Weights
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

        # Initial state
        # Set sigma to a value that covers a good portion of the remaining space
        # Since we already found a decent point, we can start slightly smaller than full box width.
        sigma = 1.5 
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky factor of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        # Stagnation tracking
        best_f_history = best_f
        no_improve_count = 0
        stagnation_threshold = 50  # Number of evaluations without improvement before restart

        # Main Loop
        while evals + lam <= budget:
            if time.time() > deadline:
                break

            # Sample lambda candidates
            Z = np.random.randn(lam, dim)
            X = m + sigma * (Z @ L.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lam

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += lam

            # Update Mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Sigma and Covariance
            # Compute C^{-1/2} v
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-18, None)
            
            v = (m_new - m) / sigma
            v_rot = eigvecs.T @ v
            v_scaled = v_rot / np.sqrt(eigvals)
            v_inv_half = eigvecs @ v_scaled
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # h_sigma
            h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_lambda) < h_sigma_thresh else 0.0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

            # Update C
            Y = (X_sorted[:mu] - m) / sigma
            M2 = (Y.T * weights) @ Y
            
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
            C = (C + C.T) / 2  # Symmetrize

            # Update m
            m = m_new

            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Reset if C becomes invalid
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                m = best_x.copy()

            # Stagnation check: if no improvement for a while, restart around best point
            if no_improve_count > stagnation_threshold:
                # Reset CMA-ES around the best point with a larger step size
                m = best_x.copy()
                sigma = max(sigma * 2.0, 1.0)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                no_improve_count = 0

        return best_x, best_f
