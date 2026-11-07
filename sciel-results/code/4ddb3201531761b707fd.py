"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        # CMA-ES Parameters
        # Population size lambda
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + 3*1 = 7. We use 10 for better exploration.
        lam = 10
        mu = lam // 2  # Number of selected individuals
        
        # Weights for selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor (mu_eff)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - c_c + min(mu_eff / (n + 2), 2)) / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start at origin, which is a good heuristic for BBOB instances often centered near 0
        # Or use a random start if origin is bad, but origin is safe.
        mean = np.zeros(n)
        sigma = 2.0  # Initial step size. Box is [-5, 5], so 2.0 covers a good range.
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        # Helper for matrix square root inversion
        def _compute_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small eigenvalues to avoid numerical issues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = _compute_sqrt_inv(C)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = mean.copy()
        best_f = np.inf
        calls = 0
        
        # Evaluate initial mean
        f_mean = func(mean)
        calls += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = mean.copy()

        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample y ~ N(0, I)
                y = rng.standard_normal(n)
                # Transform to x
                x = mean + sigma * (C_sqrt @ y)
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls += 1
                
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(ys) < lam:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]
            
            mean_old = mean
            mean = mean + sigma * (C_sqrt @ y_w)
            mean = np.clip(mean, -5, 5)
            
            # Update sigma (step size)
            # Calculate norm of y_w in the transformed space
            # ||y_w|| / ||N(0, I)||
            h_sigma = np.linalg.norm(y_w) / np.sqrt(1 - np.exp(-2.0 * (1 + c_sigma) / (n + 1)))
            h_sigma = min(h_sigma, 1.5 + 2 / (n + 1))
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update C
            # Outer product of p_c
            C = (1 + c_1 * (c_mu + c_c * (2 - c_c)) - c_1) * C
            C += c_1 * np.outer(p_c, p_c)
            
            # Weighted sum of outer products of selected y's
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Recalculate square root and inverse
            C_sqrt, C_sqrt_inv = _compute_sqrt_inv(C)
            
            # Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
