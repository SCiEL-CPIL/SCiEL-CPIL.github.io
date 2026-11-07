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
        
        n_calls = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        # Multi-restart CMA-ES with aggressive restart policy
        # Each restart uses a small budget; restart when stalled
        n_max_restarts = 8
        budget_per_restart = max(50, budget // n_max_restarts)
        
        rng = np.random.RandomState(123)
        
        for restart in range(n_max_restarts):
            if n_calls >= budget:
                break
            if time.time() > deadline - 15:
                break
            
            remaining = budget - n_calls
            this_budget = min(budget_per_restart, remaining)
            if this_budget < 20:
                break
            
            # Initialize: spread starts across the domain
            if restart == 0:
                x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                sigma = 2.0
            elif restart == 1:
                x = np.array([1.5, -1.5, 1.5, -1.5, 1.5])
                sigma = 1.5
            elif restart == 2:
                x = np.array([-2.0, 2.0, -2.0, 2.0, -2.0])
                sigma = 1.5
            elif restart == 3:
                x = np.array([2.5, 2.5, -2.5, -2.5, 2.5])
                sigma = 1.0
            elif restart == 4:
                x = np.array([-2.5, -2.5, 2.5, 2.5, -2.5])
                sigma = 1.0
            elif restart == 5:
                x = best_x + rng.randn(dim) * 0.5
                sigma = 0.8
            elif restart == 6:
                x = best_x + rng.randn(dim) * 0.3
                sigma = 0.5
            else:
                x = best_x + rng.randn(dim) * 0.2
                sigma = 0.3
            
            # CMA-ES with small population
            lam = 4
            mu = 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
            
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            DC = np.linalg.cholesky(C)
            
            # Evaluate initial point
            fx = func(x)
            n_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            prev_best = best_fx
            no_improve_gen = 0
            max_no_improve = 4
            
            while n_calls < budget:
                if time.time() > deadline - 15:
                    break
                
                # Generate population
                Z = rng.randn(lam, dim)
                Y = Z @ DC.T
                
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
                
                if n_calls >= budget:
                    break
                
                # Sort
                indices = np.argsort(F)
                Z_sorted = Z[indices]
                Y_sorted = Y[indices]
                
                # Update mean
                z_bar = np.dot(weights, Z_sorted[:mu])
                mean_step = DC @ z_bar
                x_new = x + sigma * mean_step
                
                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
                
                # Update sigma
                chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
                
                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
                
                # Update C
                h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
                Y_sel = Y_sorted[:mu]
                M = np.zeros((dim, dim))
                for i in range(mu):
                    M += weights[i] * np.outer(Y_sel[i], Y_sel[i])
                
                C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
                C = (C + C.T) / 2
                
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 0.1 * np.eye(dim)
                    try:
                        DC = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        DC = np.eye(dim)
                
                x = x_new
                
                # Check for stagnation
                if best_fx < prev_best - 1e-10:
                    prev_best = best_fx
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1
                
                if no_improve_gen >= max_no_improve:
                    break
        
        # Final NM polish
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                max_nm_calls = min(remaining - 1, 80)
                result = minimize(
                    func, best_x.copy(), method='Nelder-Mead',
                    options={'maxfun': max_nm_calls, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
