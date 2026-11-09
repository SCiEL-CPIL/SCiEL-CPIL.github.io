import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 290.0 # Safety margin for 300s limit
        
        # Initialize
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1
        
        # CMA-ES Implementation
        # Parameters
        popsize = 4 + int(3 * np.log(self.dim))
        if popsize < 5: popsize = 5
        if popsize > self.budget // 10: popsize = self.budget // 10
        
        sigma = 2.0
        C = np.eye(self.dim)
        Cs = np.zeros((self.dim, self.dim))
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Update rates
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        sigma_suc = (1 + 2 * max(0, cs - 1)) * (1 + 2 * min(0, cc - 1) * mueff / (self.dim + 2))
        
        chi_n = np.sqrt(self.dim) * (1 - 1. / (4 * self.dim) + 1. / (21 * self.dim**2))
        
        # Main Loop
        while calls < self.budget:
            if time.time() - start_time > deadline:
                break
                
            # Generate offspring
            X = np.zeros((popsize, self.dim))
            F = np.zeros(popsize)
            
            for i in range(popsize):
                z = rng.standard_normal(self.dim)
                x_i = best_x + sigma * (np.linalg.cholesky(C) @ z)
                x_i = np.clip(x_i, -5, 5)
                X[i] = x_i
                F[i] = func(x_i)
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
                
                if calls >= self.budget:
                    break
            
            if calls >= self.budget:
                break
            
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx[:mu]]
            F_sorted = F[idx[:mu]]
            
            # Update mean
            m_old = best_x
            best_x = np.sum(weights[:, np.newaxis] * X_sorted, axis=0)
            
            # Update path
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) / np.linalg.cholesky(C) @ (best_x - m_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.cholesky(C) @ (best_x - m_old) / sigma
            
            # Update Covariance
            delta_sig = (np.linalg.norm(ps) / chi_n - 1)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1 + 2 / self.dim) * chi_n
            hsig = 1 if hsig else 0
            
            C = (1 + cc * (2 - cc) * hsig - cc) * C + \
                cc * (2 - cc - hsig) * np.outer(pc, pc) + \
                (1 - (1 + cc * (2 - cc) * hsig - cc)) * np.sum(
                    weights[:, np.newaxis, np.newaxis] * np.outer(X_sorted - m_old, X_sorted - m_old) / sigma**2, 
                    axis=0
                )
                
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Add small identity to ensure PD if needed, though CMA-ES usually handles it
            # Simple check: if eigenvalues become negative, reset C to identity? 
            # For robustness, we can just clip eigenvalues or add epsilon.
            # Let's add a small epsilon to diagonal for numerical stability
            C += 1e-8 * np.eye(self.dim)
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Prevent sigma from becoming 0 or inf
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 100:
                sigma = 100

        # Final Polish with Nelder-Mead if budget allows
        remaining = self.budget - calls
        if remaining > 50 and time.time() - start_time < deadline - 10:
            try:
                res = so.minimize(func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': remaining // 10, 'xatol': 1e-6, 'fatol': 1e-6})
                # Note: minimize will use up to maxiter * dim evals roughly.
                # We don't strictly control the exact number of calls in minimize, 
                # but maxiter=remaining//10 is a safe heuristic for 5D.
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
