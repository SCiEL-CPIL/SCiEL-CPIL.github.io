import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_pop = max(4, dim)  # population size
        mu = lambda_pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
        
        # Initial population
        for i in range(lambda_pop):
            if not time_ok():
                break
            z = np.random.randn(dim)
            x_i = np.clip(x_mean + sigma * C_sqrt @ z, -5, 5)
            f_i = func(x_i)
            calls += 1
            if f_i < best_f:
                best_f = f_i
                best_x = x_i.copy()
        
        # Main loop
        while calls < budget and time_ok():
            # Generate population
            zs = np.random.randn(lambda_pop, dim)
            xs = np.clip(x_mean + sigma * (zs @ C_sqrt.T), -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            calls += lambda_pop
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Weighted mean
            x_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # Update C and sigma
            x_diff = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * x_diff
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lambda_pop))) < (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * x_diff
            
            C = (1 + c_1 + c_mu - c_1 * np.trace(C)) * C + c_1 * np.outer(x_diff, x_diff) + c_mu * np.sum(w[:, None, None] * np.outer(xs_sorted[:mu] - x_mean, xs_sorted[:mu] - x_mean) / sigma**2, axis=0)
            
            # Ensure C is PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt((dim + 1) / (dim - 1)) - 1))
            
            x_mean = x_new
        
        return best_x, best_f
