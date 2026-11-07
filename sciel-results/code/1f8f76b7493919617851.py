import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        if lam < 2: lam = 2
        mu = max(1, lam // 2)
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control
        c_sigma = 1.0 / lam
        d_sigma = 1.0 + 2.0 * max(0.0, (mueff - 1.0) / (lam + 1.0)) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (5.0 + lam / 2.0) / (lam + 5.0 + 2.0)
        c_1 = 2.0 / ((lam + 1.3)**2 + n)
        c_mu = (mueff - 2.0 + 1.0 / c_1) / ((lam - mueff + 1.3)**2 + n)
        c_mu = min(c_mu, 2.0 * (1.0 - c_1) / (n - 1.0))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation of the mean? 
        # CMA-ES usually doesn't evaluate the mean, only the population.
        # But evaluating the mean can help if it's good.
        # However, it costs 1 call.
        # Let's stick to standard CMA-ES which evaluates lambda points per generation.
        
        calls_made = 0
        
        def time_exceeded():
            return time.time() > deadline

        max_gens = budget // lam
        
        for g in range(max_gens):
            if time_exceeded():
                break
            if calls_made + lam > budget:
                # Not enough budget for a full generation
                # Do a few single steps? Or just break.
                # Breaking is safer to avoid partial generation logic complexity.
                break
                
            # Update inverse square root of C
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                break

            # Generate population
            ys = []
            for i in range(lam):
                if time_exceeded() or calls_made >= budget:
                    break
                z = np.random.randn(n)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                ys.append(y)
                calls_made += 1
                
            if len(ys) < lam:
                break

            # Evaluate
            fs = []
            for y in ys:
                if time_exceeded() or calls_made >= budget:
                    break
                f = func(y)
                fs.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = y.copy()
            
            if len(fs) < lam:
                break
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Weighted recombination
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * ys_sorted[i]
                
            # Update mean
            x_mean_new = x_mean + c_c * (y_w - x_mean)
            
            # Update step size
            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * (C_inv_sqrt @ (y_w - x_mean)) / sigma
            
            hs = np.sqrt(np.sum((C_inv_sqrt @ (C_inv_sqrt.T @ ps))**2))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
            
            sigma *= np.exp(c_sigma * d_sigma * (hs / chi_n - 1.0))
            
            # Update covariance matrix
            pc = (1.0 - c_c) * pc + np.sqrt(c_c * (2.0 - c_c) * mueff) * (y_w - x_mean) / sigma
            
            C = (1.0 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1.0 - c_c) * (2.0 - c_c) * C)
            
            v = np.array(ys_sorted[:mu]) - x_mean
            v = v / sigma
            rank_mu = np.sum([w[i] * np.outer(v[i], v[i]) for i in range(mu)], axis=0)
            C += c_mu * rank_mu
            
            x_mean = x_mean_new

        return best_x, best_f
