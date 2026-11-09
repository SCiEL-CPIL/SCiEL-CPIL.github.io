import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES setup
        popsize = 4 + int(3 * np.log(n))
        mu = popsize // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        
        x_mean = np.random.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute matrices
        try:
            sqrtC = np.linalg.cholesky(C)
            invsqrtC = np.linalg.inv(sqrtC)
        except:
            sqrtC = np.eye(n)
            invsqrtC = np.eye(n)
            
        best_x = x_mean
        best_f = func(x_mean)
        calls = 1
        gen = 0
        
        while calls < budget:
            # Generate population
            # Ensure we don't exceed budget
            if budget - calls < popsize:
                # If we can't do a full generation, do a partial one or break?
                # Doing a partial one with fewer points is okay for mean update,
                # but covariance update needs care.
                # For simplicity and robustness, if we can't do a full gen,
                # we do a simple (1+1) step with remaining budget.
                # Or just break. Given 500/8 = 62 rem 4, we lose 4 calls.
                # 4/500 is 0.8%. Acceptable.
                break
            
            X = np.zeros((popsize, n))
            for i in range(popsize):
                z = np.random.randn(n)
                x_new = x_mean + sigma * (sqrtC @ z)
                X[i] = np.clip(x_new, -5, 5)
                calls += 1
            
            F = np.array([func(x) for x in X])
            
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.dot(w[:mu], X_sorted[:mu])
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (invsqrtC @ (x_mean - x_old) / sigma)
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            # hsig calculation
            # hsig = 1 if norm_ps < (1 + 2/(n+1)) * sqrt((n+1) * (1 - (1-2*gen/popsize)^(2*cc))) * sqrt(1 - (1-2*gen/popsize)^(2*cs)) else 0
            # Simplified hsig for early gens
            hsig = 1 if norm_ps < 1.4 * np.sqrt(n) else 0
            
            sigma = sigma * np.exp(cs / mueff * (norm_ps / np.sqrt(mueff) - 1))
            
            # Update pc
            pc = (1 - cc) * pc + cc * (x_mean - x_old) / sigma
            
            # Update C
            # delta_h
            delta_h = 1 + 2 * max(0, np.sqrt(cc / (1 - cc)) - 1) * (1 - hsig)
            
            # Rank-1 update
            p_sigma = (x_mean - x_old) / sigma
            # Rank-mu update
            # Steps: (X_sorted[:mu] - x_old) / sigma
            steps = (X_sorted[:mu] - x_old) / sigma
            rank_mu = np.sum(w[:mu]**2) * (steps @ steps.T)
            
            C = (1 + c1 * delta_h + cmu - c1) * C \
                + c1 * (p_sigma @ p_sigma.T - C) \
                + cmu * rank_mu
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure PD
            try:
                sqrtC = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(sqrtC)
            except:
                C += 1e-8 * np.eye(n)
                sqrtC = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(sqrtC)
                
            gen += 1
            
        return best_x, best_f
