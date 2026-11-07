import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        
        # CMA-ES Parameters
        lam = 15  # Population size
        sigma = 1.0
        m = np.zeros(dim)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        calls = 0
        x_best = m.copy()
        f_best = np.inf
        
        def eval_func(x):
            nonlocal calls, x_best, f_best
            if calls >= budget:
                return np.inf
            fx = func(x)
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Initial evaluation
        f_best = eval_func(m)
        
        while calls < budget and time.time() < deadline:
            ys = []
            xs = []
            fs = []
            
            for _ in range(lam):
                if calls >= budget:
                    break
                z = np.random.randn(dim)
                x = m + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = eval_func(x)
                ys.append(z)
                xs.append(x)
                fs.append(fx)
                
            if calls >= budget:
                break
                
            if len(xs) < 2:
                break

            order = np.argsort(fs)
            ys = np.array([ys[i] for i in order])
            xs = np.array([xs[i] for i in order])
            fs = np.array([fs[i] for i in order])
            
            m_old = m.copy()
            m = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            y_bar = np.sum(weights[:, None] * ys[:mu], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_bar)
            cs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lam)))
            
            pc = (1 - c_c) * pc + c_c * (m - m_old) / sigma
            
            delta_d = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) * (cs - 1)
            C = (1 + c_c * (2 - c_c) * delta_d) * C - c_c * np.outer(pc, pc)
            
            C += (1 - c_c) * c_sigma * mu_eff * np.sum(weights[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
            
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) * (1 - c_sigma / (2 * d_sigma)) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            if np.max(fs[:mu]) - np.min(fs[:mu]) < 1e-12:
                break

        return x_best, f_best
