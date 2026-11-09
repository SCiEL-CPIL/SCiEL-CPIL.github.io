import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Increase population size for better convergence in low dim
        lam = 10
        if lam > budget:
            lam = max(2, budget)
            
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt(mueff / n) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Initial state
        m = np.zeros(n)
        # Larger initial sigma to cover the box better
        sigma = 3.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Pre-compute B, D
        B = np.eye(n)
        D = np.ones(n)
        
        gen = 0
        # Reserve some budget for local search
        local_search_budget = 50
        cma_budget = max(lam, budget - local_search_budget)
        
        while calls_made + lam <= cma_budget:
            gen += 1
            # Sample
            Y = np.random.randn(lam, n)
            P = m + sigma * (Y @ B.T)
            P = np.clip(P, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in P])
            calls_made += lam
            
            idx = np.argmin(F)
            if F[idx] < best_f:
                best_f = F[idx]
                best_x = P[idx].copy()
            
            # Sort
            idx_sorted = np.argsort(F)
            Y_sorted = Y[idx_sorted]
            
            # Update mean
            m_old = m.copy()
            m = m + sigma * np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update ps, pc
            y_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ y_mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen))) < (1.4 + 2 / (n + 1)) * chiN
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (1 - cc) * m_old @ m_old.T) + cmu * np.sum(weights[:, None, None] * (Y_sorted[:mu] - y_mean)[:, :, None] * (Y_sorted[:mu] - y_mean)[:, None, :], axis=0)
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update B, D
            try:
                C_cholesky = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                B = C_cholesky.T
                D = np.diag(C_cholesky)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                sigma *= 0.5
            
            if sigma < 1e-10 or sigma > 1e10:
                break

        # Use remaining budget for local search
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None:
            # Use Nelder-Mead for local refinement
            try:
                res = scipy.optimize.minimize(func, best_x, method='Nelder-Mead', 
                                              options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
