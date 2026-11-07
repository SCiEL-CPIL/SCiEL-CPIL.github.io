import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 8
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Mu_eff: effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping of covariance update
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        # Initial state
        x = np.zeros(dim)
        sigma = 1.5
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Cholesky factor
        DC = np.linalg.cholesky(C)
        
        # Evaluation counter
        n_calls = 0
        
        # Initial evaluation
        fx = func(x)
        n_calls += 1
        
        best_x = x.copy()
        best_fx = fx
        
        # Main loop
        while n_calls < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate population
            Z = np.random.randn(lam, dim)
            Y = Z @ DC.T
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                xi = x + sigma * Y[i]
                F[i] = func(xi)
                n_calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = xi.copy()
            
            # If we broke out of the eval loop early (budget exhausted), break main loop
            if n_calls >= budget:
                break
            
            # Sort offspring by fitness
            indices = np.argsort(F)
            Z_sorted = Z[indices]
            Y_sorted = Y[indices]
            
            # Update mean
            w = weights
            z_bar = np.dot(w, Z_sorted[:mu])
            mean_step = DC @ z_bar
            x_new = x + sigma * mean_step
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            
            # Update sigma
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
            
            # Update C
            # h_sig is the indicator for the sigma update being valid
            h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
            
            # Rank-mu update
            Y_sel = Y_sorted[:mu]
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += w[i] * np.outer(Y_sel[i], Y_sel[i])
            
            # Update C
            C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
            
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            
            # Check positive definiteness
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add diagonal
                C += 0.1 * np.eye(dim)
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    DC = np.eye(dim)
            
            x = x_new
        
        # Nelder-Mead polish with remaining budget
        remaining = budget - n_calls
        if remaining > 20 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                max_nm_calls = min(remaining, 200)
                result = minimize(
                    func, best_x.copy(), method='Nelder-Mead',
                    options={'maxiter': max_nm_calls, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': max_nm_calls, 'adaptive': True}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
