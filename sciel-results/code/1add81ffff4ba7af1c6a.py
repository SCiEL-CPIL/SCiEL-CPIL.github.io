"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam < 2:
            lam = 2
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff)))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        # State
        x_mean = rng.uniform(-5, 5, d)  # Random start
        sigma = 1.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        # Pre-compute Cholesky of C (initially identity)
        # We will update C every generation
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            remaining = budget - calls
            current_lam = min(lam, remaining)
            
            if current_lam <= 0:
                break
            
            # Generate population
            Y = []
            F = []
            
            # We need the square root of C. 
            # For efficiency, we can use Cholesky decomposition.
            # Since d=5, this is very fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(d)
                L = np.eye(d)
            
            for i in range(current_lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (L @ z)
                # Clip to box? The problem says func clips, but keeping y in bounds 
                # helps avoid wasted evaluations if the function is undefined outside 
                # (though contract says it clips). We clip to be safe and efficient.
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                Y.append(y)
                F.append(fy)
            
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            
            # Determine effective mu for this generation
            current_mu = min(mu, current_lam)
            
            # Update mean
            x_old = x_mean.copy()
            # Note: weights are normalized for original mu. If current_mu < mu, 
            # we should ideally renormalize weights or just use the available ones.
            # For simplicity and robustness, if current_mu < mu, we can renormalize 
            # the first current_mu weights.
            if current_mu < mu:
                w_sum = np.sum(weights[:current_mu])
                w_eff = weights[:current_mu] / w_sum
            else:
                w_eff = weights[:current_mu]
                
            for i in range(current_mu):
                x_mean += w_eff[i] * Y[i]
            
            # Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * (C^-0.5 * (x_new - x_old) / sigma)
            # C^-0.5 is inverse of L^T. Since L is lower triangular, L^T is upper.
            # Solve L^T v = (x_new - x_old) / sigma
            try:
                diff = (x_mean - x_old) / sigma
                v = np.linalg.solve(L.T, diff)
            except np.linalg.LinAlgError:
                v = diff
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1 + 2 / (d + 1)) * np.sqrt(1 + c_sigma)
            
            if h_sigma:
                p_sigma = p_sigma # No update to p_sigma effectively? No, p_sigma is updated, but c_sigma is not applied to C?
                # Standard CMA-ES: if h_sigma == 0, p_sigma is not used to update C?
                # Actually, h_sigma is a condition for updating C using p_sigma.
                pass
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c + (1 - h_sigma) * c_sigma * (2 - c_sigma) * p_sigma * p_sigma) + c_mu * sum w_i * (y_i - x_old) * (y_i - x_old)^T
            
            # First update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * c_c * sqrt(mu_eff) * C^-0.5 * (x_new - x_old)
            # We already have v = C^-0.5 * (x_new - x_old) / sigma
            # So C^-0.5 * (x_new - x_old) = sigma * v
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * sigma * v
            
            # Calculate rank-one update term
            # Note: h_sigma is boolean 0 or 1
            h_sigma_float = 1.0 if h_sigma else 0.0
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma_float) * c_sigma * (2 - c_sigma) * np.outer(p_sigma, p_sigma))
            
            # Rank-mu update
            for i in range(current_mu):
                diff = Y[i] - x_old
                C += c_mu * w_eff[i] * np.outer(diff, diff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) - np.sqrt(d)))
            
            # Ensure sigma doesn't get too small or large
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Termination check
            if sigma < 1e-6:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
