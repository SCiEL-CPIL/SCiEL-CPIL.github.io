import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 290.0  # Safety margin

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))  # Population size
        mu = lam // 2  # Number of points used for update
        
        # Initial state
        x0 = np.zeros(n)
        # Try to start near 0, but if 0 is bad, CMA-ES will move.
        # Alternatively, start with a random point if we want to be safer against shifts?
        # The problem says minimizer is in [-4, 4]. 0 is a good bet.
        
        mean = x0.copy()
        cov = np.eye(n)
        sigma = 1.0  # Initial step size
        
        # Learning rates
        c_sigma = (mu + 2) / (n + mu + 5)
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 0.5) / (n + 2)) ** 2 - c_1)
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Path variables for sigma adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = mean.copy()
        best_fx = float('inf')
        
        try:
            # Evaluate initial point
            best_fx = eval_func(mean)
            best_x = mean.copy()
            
            while self.calls < self.budget and time.time() - self.start_time < self.time_limit:
                # Generate population
                points = []
                for i in range(lam):
                    # Sample from N(0, cov)
                    # Use Cholesky decomposition for efficiency and stability
                    L = np.linalg.cholesky(cov)
                    z = np.random.randn(n)
                    y = L @ z
                    x_new = mean + sigma * y
                    points.append(x_new)
                
                # Evaluate population
                f_vals = []
                for x in points:
                    f_vals.append(eval_func(x))
                
                # Sort by fitness
                indices = np.argsort(f_vals)
                points = [points[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]
                
                # Update best
                if f_vals[0] < best_fx:
                    best_fx = f_vals[0]
                    best_x = points[0].copy()
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * points[i]
                
                # Update covariance
                # Calculate deviations
                diffs = [points[i] - mean for i in range(mu)]
                new_cov = np.zeros((n, n))
                for i in range(mu):
                    new_cov += weights[i] * np.outer(diffs[i], diffs[i])
                
                # Correction terms for CMA-ES (simplified)
                # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(n+1)) * sqrt((mu_eff+1)/mu_eff - 1) * c_sigma
                # For simplicity and robustness, we might skip the full h_sigma logic 
                # and use a simpler step size adaptation or standard CMA-ES rules.
                # Given the short budget, a simpler adaptation might be more stable.
                
                # Simple sigma adaptation:
                # If progress is slow, increase sigma? No, usually decrease if stuck.
                # Standard CMA-ES sigma update:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (new_mean - mean)
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (self.calls // lam + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / mu_eff - 1) * c_sigma
                
                if h_sigma:
                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
                    delta_h = 1 - (1 - c_c) ** (self.calls // lam + 1)
                    new_cov = (1 - c_1 - c_mu) * cov + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * cov) + c_mu * np.sum([weights[i] * np.outer(diffs[i], diffs[i]) for i in range(mu)], axis=0)
                else:
                    p_c = (1 - c_c) * p_c
                    delta_h = 1 - (1 - c_c) ** (self.calls // lam + 1)
                    new_cov = (1 - c_1 - c_mu) * cov + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * cov) + c_mu * np.sum([weights[i] * np.outer(diffs[i], diffs[i]) for i in range(mu)], axis=0)
                
                # Ensure positive semi-definiteness
                # Symmetrize
                new_cov = (new_cov + new_cov.T) / 2
                # Add small regularization if needed
                new_cov += 1e-10 * np.eye(n)
                
                # Update sigma
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (self.calls // lam + 1)) - np.sqrt((1 - (1 - c_sigma) ** 2) * (mu_eff - 1))))
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                mean = new_mean
                cov = new_cov
                
        except StopIteration:
            pass
        except Exception as e:
            # If CMA-ES fails, fall back to a simple local search or just return best
            pass

        # Final polish with Nelder-Mead or Powell if budget remains
        remaining = self.budget - self.calls
        if remaining > 10 and time.time() - self.start_time < self.time_limit - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(eval_func, best_x, method='Powell', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
