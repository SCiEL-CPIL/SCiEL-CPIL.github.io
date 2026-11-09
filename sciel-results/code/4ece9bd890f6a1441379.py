import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_w = np.sum(weights**2)
        
        c_sigma = (mu_w + 2) / (n + mu_w + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (n + 1)) - 1) + c_sigma
        
        c_c = (4 + mu_w / n) / (n + 4 + 2 * mu_w / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_w)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((n + 2)**2 + mu_w))
        
        # Initial state
        # Start at origin, which is likely close to the shifted optimum
        x_mean = np.zeros(n)
        sigma = 1.0
        
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1
        
        L = np.linalg.cholesky(C)
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
                
            ys = []
            xs = []
            fs = []
            
            for j in range(lam):
                z = np.random.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y_clipped = np.clip(y, -5, 5)
                
                fy = func(y_clipped)
                calls += 1
                
                ys.append(z)
                xs.append(y_clipped)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y_clipped.copy()
            
            idx = np.argsort(fs)
            ys = np.array(ys)[idx]
            xs = np.array(xs)[idx]
            fs = np.array(fs)[idx]
            
            x_mean_new = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
            
            diff = (x_mean_new - x_mean) / sigma
            v = np.linalg.solve(L, diff)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * v
            
            w = np.linalg.solve(L, v)
            u = np.linalg.solve(L.T, w)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_w) * u
            
            x_mean = x_mean_new
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (i + 1))) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C
            C += c_c * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(n))
            
            for k in range(mu):
                C += c_mu * weights[k] * np.outer(ys[k], ys[k])
            
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                p_c = np.zeros(n)
                p_sigma = np.zeros(n)

        return best_x, best_f
