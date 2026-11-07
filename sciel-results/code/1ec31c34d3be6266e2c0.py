"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize CMA-ES parameters
        # Start near the center of the box, as the minimizer is in [-4, 4]^5
        x = np.zeros(dim)
        sigma = 2.0
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        def update_sqrt(C, C_sqrt, C_sqrt_inv):
            """Update square root and inverse square root of C."""
            D, V = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            D_sqrt = np.sqrt(D)
            D_sqrt_inv = 1.0 / D_sqrt
            C_sqrt = V @ np.diag(D_sqrt) @ V.T
            C_sqrt_inv = V @ np.diag(D_sqrt_inv) @ V.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = update_sqrt(C, C_sqrt, C_sqrt_inv)
        
        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        
        while calls < budget:
            # Check time limit
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                
            # Evaluate
            fvals = np.array([func(y) for y in ys])
            calls += lambda_
            
            # Sort
            idx = np.argsort(fvals)
            ys = [ys[i] for i in idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = ys[0].copy()
                
            # Mean update
            x_new = sum(w[i] * ys[i] for i in range(mu))
            
            # Step size control
            d = x_new - x
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d_hat = d / d_norm
            else:
                d_hat = np.zeros(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ d_hat)
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) / chi_n) < 1.4 + 2 / (dim + 1) else 0.0
            
            # Position update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_hat
            x = x_new
            
            # Covariance update
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(w[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))
            
            # Ensure positive definite
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update square roots
            C_sqrt, C_sqrt_inv = update_sqrt(C, C_sqrt, C_sqrt_inv)
            
            # Sigma update
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
