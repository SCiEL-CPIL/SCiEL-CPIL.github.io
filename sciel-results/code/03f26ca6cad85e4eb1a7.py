import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.max_time = 290.0  # Safety margin for 300s limit

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        if lam < 6: lam = 6
        if lam > 20: lam = 20
        
        # Initial state
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Parameters
        cmu = 2 / ((n + 1.3) ** 2 + lam)
        c_sigma = (1 + 2 * max(0, np.sqrt((lam / 2 - 1)) / (n + 2) - 1)) / (n + 4)
        c_c = (4 + lam / n) / (n + 4 + 2 * lam / n)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Update rates
        c_cov_pos = (2 + 2 * np.sqrt(mu_eff / (n + 2))) / (n + 3)
        c_cov_neg = (2 + 2 * np.sqrt((mu_eff - 2) / (n + 2))) / (n + 5)
        
        # Initial evaluation
        x_clipped = np.clip(x, -5, 5)
        f_best = func(x_clipped)
        calls = 1
        best_x = x_clipped.copy()
        
        # Main loop
        while calls + lam <= budget:
            # Time check
            if time.time() - self.start_time > self.max_time:
                break
            
            # Sample lambda points
            z = np.random.randn(lam, n)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            y = z @ L.T
            xs = np.array([x + sigma * y[i] for i in range(lam)])
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(xi) for xi in xs])
            calls += lam
            
            # Sort
            idx = np.argsort(fs)
            fs = fs[idx]
            xs = xs[idx]
            
            # Update best
            if fs[0] < f_best:
                f_best = fs[0]
                best_x = xs[0].copy()
            
            # Update mean
            x_new = np.sum(xs[:mu, :] * weights[:, np.newaxis], axis=0)
            
            # Update evolution paths
            d = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lam)) < (1.4 + 2 / (n + 1)) * chi_n) else 0
            y_sigma = p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lam))
            
            C = (1 + c_cov_pos * h_sigma * (mu_eff - 2) - c_cov_neg) * C
            C += c_cov_pos * np.outer(y_sigma, y_sigma)
            C += c_cov_neg * np.sum([np.outer(xs[i] - x, xs[i] - x) / sigma ** 2 for i in range(mu)] * weights, axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            x = x_new
            
            # Check convergence
            if sigma < 1e-10:
                break
            if np.all(np.abs(xs[:mu] - x) < 1e-8):
                break

        return best_x, f_best
