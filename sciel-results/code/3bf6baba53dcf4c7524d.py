import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        def clip(v):
            return np.clip(v, -5.0, 5.0)
        
        def time_ok():
            return time.time() < deadline and total_evals < budget

        # Phase 1: Coarse multi-start
        n_starts = 10
        candidates = rng.uniform(-4.0, 4.0, size=(n_starts, dim))
        # Add origin
        candidates = np.vstack([candidates, np.zeros(dim)])
        
        for i in range(len(candidates)):
            if not time_ok():
                break
            x = clip(candidates[i])
            fx = func(x)
            total_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            x = clip(rng.uniform(-4, 4, dim))
            best_f = func(x)
            total_evals += 1
            best_x = x.copy()

        # Phase 2: CMA-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # CMA-ES parameters for dim=5
        lambda_ = 4 * dim
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1.0 / mu_eff) / (dim + 2) ** 2 - 1))
        
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        C = np.eye(dim) # Initialize covariance matrix

        while time_ok() and total_evals + lambda_ <= budget:
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if not time_ok() or total_evals >= budget:
                    break
                y = clip(x + sigma * (B @ (D * rng.standard_normal(dim))))
                fy = func(y)
                total_evals += 1
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) < lambda_:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            new_x = sum(w * y for w, y in zip(weights, ys[:mu]))
            
            # Update covariance matrix
            diffs = np.array(ys[:mu]) - x
            # Rank-one update
            z = p_c / np.sqrt(c_c * (1 - c_c))
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(z, z) + (1 - c_c) * np.eye(dim))
            C += c_mu * sum(w * np.outer(d, d) for w, d in zip(weights, diffs))
            
            # Update step size control vector
            y_sigma = (new_x - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_sigma
            sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (1 - c_c) * mu_eff) * (B @ y_sigma)
            
            # Update B and D (eigen decomposition)
            B, D_sq = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D_sq, 1e-12))
            
            x = new_x
            fx = fs[0]

        return best_x, best_f
