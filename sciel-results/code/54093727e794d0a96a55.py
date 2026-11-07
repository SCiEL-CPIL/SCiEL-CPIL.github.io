"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for 5-dimensional BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        # Selection size
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance reduction factor
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        # Mean update parameters
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial parameters
        x = np.zeros(d)  # Start at center
        sigma = 2.0      # Initial step size
        B = np.eye(d)    # Eigenvectors
        D = np.ones(d)   # Eigenvalues
        C = np.eye(d)    # Covariance matrix
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls = 0
        best_x = x.copy()
        best_f = np.inf
        
        def safe_call(x_in):
            nonlocal calls
            x_clip = np.clip(x_in, -5, 5)
            f_val = func(x_clip)
            calls += 1
            return f_val, x_clip

        # Initial evaluation
        f0, x0 = safe_call(x)
        best_f = f0
        best_x = x0
        
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                # Sample
                z = np.random.randn(d)
                # Transform
                y = x + sigma * (B @ (D * z))
                fy, y_clip = safe_call(y)
                ys.append(y_clip)
                fs.append(fy)
                
                # Track best
                if fy < best_f:
                    best_f = fy
                    best_x = y_clip
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in indices])
            fs_sorted = np.array([fs[i] for i in indices])
            
            # Update mean
            x_old = x.copy()
            x_new = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update step size
            y_sigma = (x_new - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian norm approximation for chi-square correction
            chi_bar = np.sqrt(d) * (1 + 1/(2*d) - 1/(4*d**2))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(d+1)) * chi_bar) else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_bar - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            rank_mu = np.outer(weights, np.ones((mu, d)))
            rank_mu_mat = np.zeros((d, d))
            for i in range(mu):
                diff = ys_sorted[i] - x_old
                rank_mu_mat += weights[i] * np.outer(diff, diff)
            
            rank_one_mat = np.outer(p_c, p_c)
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            C = (1 + c_1 * (c_1 - c_mu + delta_h) - c_mu) * C + c_1 * rank_one_mat + c_mu * rank_mu_mat
            
            # Recalculate B and D from C (Cholesky or Eigendecomposition)
            # Use eigendecomposition for stability
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-18))
            
            x = x_new
            
            # Periodic recentering if stuck?
            # For now, rely on CMA-ES dynamics.
            
            # If sigma becomes very small, we might be stuck.
            # But with 500 budget, we just let it run.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
