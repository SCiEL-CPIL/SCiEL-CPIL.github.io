import numpy as np


class Optimizer:
    """CMA-ES Optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        # For dim=5, lam = 4 + 4 = 8. Let's use 10 for better coverage.
        lam = max(lam, 10)
        
        # Selection
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective selection mass
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = 4.0 / (dim + 4.0)
        c_1 = 2.0 / (dim**1.8 + 10.0)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) / (dim**2 + 18.0))
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, box is [-5, 5]
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        
        # Cholesky factor of C
        # We need to update C and its Cholesky factor.
        # For small dim, we can just use np.linalg.cholesky each step or maintain it.
        # Since dim=5, recomputing Cholesky is cheap.
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        budget -= 1
        
        if budget <= 0:
            return best_x, best_f
            
        # Main loop
        while budget >= lam:
            # Generate population
            ys = np.zeros((lam, dim))
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)
            
            for i in range(lam):
                # Sample y from N(0, C)
                # y = L @ z, where L is Cholesky factor of C
                # We maintain C explicitly.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset to identity
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                z = np.random.randn(dim)
                y = L @ z
                x = x_mean + sigma * y
                
                # Clip to bounds
                x = np.clip(x, -5.0, 5.0)
                
                # Evaluate
                f_val = func(x)
                budget -= 1
                
                ys[i] = y
                xs[i] = x
                fs[i] = f_val
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                    
            if budget <= 0:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update evolution paths
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * (x_mean - x_mean_old) / sigma
            pc = (1.0 - c_c) * pc + c_c * np.sqrt(mu_eff) * (x_mean - x_mean_old) / sigma
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv @ ((x_mean - x_mean_old) / sigma)
            # Need C_inv. For dim=5, we can invert C.
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
                
            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (C_inv @ ((x_mean - x_mean_old) / sigma))
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1.0))
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - c_c) * (1 - c_c) * C) + c_mu * sum(weights * (y_i @ y_i.T))
            # Note: The term (1 - c_c) * (1 - c_c) * C is an approximation for the expected outer product of pc.
            # More accurately: C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - c_c**2) * C) + c_mu * H
            # where H = sum(weights * (y_i @ y_i.T))
            
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
                
            C = (1.0 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1.0 - c_c**2) * C) + c_mu * H
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
        return best_x, best_f
