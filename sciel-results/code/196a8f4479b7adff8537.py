"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        # Function to check time
        def time_ok():
            return time.time() < deadline and calls < total_budget

        # Helper to evaluate a point
        def eval_point(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with a random point
        x0 = rng.uniform(-5, 5, n)
        best_f = eval_point(x0)
        best_x = x0.copy()

        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        # State variables
        m = best_x.copy()
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
        
        stagnation_counter = 0
        restarts = 0
        max_restarts = 3

        while time_ok() and calls < total_budget:
            # Check for stagnation
            if stagnation_counter > 50:
                if restarts < max_restarts:
                    # Restart around best_x with larger sigma
                    m = best_x.copy()
                    sigma = 2.0
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    stagnation_counter = 0
                    restarts += 1
                    continue
                else:
                    # If no restarts left, just do random search
                    break

            # Generate population
            Z = rng.standard_normal((lam, n))
            X = np.empty((lam, n))
            for i in range(lam):
                y_i = B @ (D * Z[i])
                X[i] = m + sigma * y_i
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if not time_ok():
                    break
                F[i] = eval_point(X[i])
            
            if calls >= total_budget or time.time() > deadline:
                break

            # Sort
            indices = np.argsort(F)
            F = F[indices]
            X = X[indices]
            
            # Check improvement
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Recombination
            m_old = m.copy()
            m = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Step size adaptation
            y = (m - m_old) / sigma
            z = B.T @ (y / D)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            hs_norm = np.linalg.norm(p_sigma)
            hs_threshold = (1.4 + 2.0 / (n + 1)) * (1 - c_sigma)
            if hs_norm / hs < hs_threshold:
                chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1))
                sigma *= np.exp(c_sigma / chs * (hs_norm / hs - 1))
            
            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            
            # h_c calculation (simplified, assume 1 for stability if complex)
            # Proper h_c:
            # E[||z||^2] = n
            # threshold = (1.4 + 2/(n+1)) * sqrt((mu_eff-1)+1)
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^{2}) < threshold
            # For simplicity and robustness, we use the standard update without h_c correction term (i.e., term is 0)
            # C = (1 - c_1) C + c_1 (y y^T)
            # This is safe.
            
            C = (1 - c_1) * C + c_1 * (np.outer(y, y))
            
            # Rank-mu update
            for i in range(mu):
                y_i = (X[i] - m_old) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))
            
            # Convergence check
            if np.max(D) / np.min(D) > 1e14 or sigma < 1e-15:
                # Restart
                m = best_x.copy()
                sigma = 2.0
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                stagnation_counter = 0
                restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
