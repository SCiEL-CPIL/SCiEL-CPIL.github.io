"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart for Lunacek bi-Rastrigin (f24)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        fbest_global = np.inf
        calls_total = 0
        
        # Run 1: CMA-ES starting at origin with moderate sigma
        run_budget = max(300, int(budget * 0.6))
        x0, sigma, best_x, fbest, calls_used = self._run_cmaes(
            func, run_budget, dim,
            start_point=np.zeros(dim),
            sigma_init=2.5
        )
        calls_total += calls_used
        if fbest < fbest_global:
            fbest_global = fbest
            best_x_global = best_x
        
        # Run 2: If budget remains, try a second CMA-ES from a kicked start
        if calls_total < budget - 30 and time.time() < deadline - 30:
            remaining = budget - calls_total
            run_budget2 = max(30, remaining)
            # Kick in random direction from best found
            kick_dir = np.random.standard_normal(dim)
            kick_dir /= np.linalg.norm(kick_dir)
            start2 = np.clip(best_x_global + 4.0 * kick_dir, -5, 5)
            x0_2, sigma_2, best_x_2, fbest_2, calls_used_2 = self._run_cmaes(
                func, run_budget2, dim,
                start_point=start2,
                sigma_init=1.5
            )
            calls_total += calls_used_2
            if fbest_2 < fbest_global:
                fbest_global = fbest_2
                best_x_global = best_x_2
        
        # Final polish: Nelder-Mead on the best point found
        if time.time() < deadline - 10 and best_x_global is not None:
            best_x_global, fbest_global = self._polish(func, best_x_global, fbest_global, dim, budget - calls_total)
        
        return best_x_global, fbest_global

    def _run_cmaes(self, func, budget, dim, start_point=None, sigma_init=1.0, restart_offset=None):
        # Parameters
        lambda_ = 15
        mu = lambda_ // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        if start_point is None:
            if restart_offset is not None:
                kick_dir = np.random.standard_normal(dim)
                kick_dir /= np.linalg.norm(kick_dir)
                x0 = restart_offset + 3.5 * kick_dir
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = np.zeros(dim)
        else:
            x0 = start_point.copy()
            
        sigma = sigma_init
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        def sqrtm(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-10)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        
        B = sqrtm(C)
        
        fbest = func(x0)
        best_x = x0.copy()
        calls = 1
        
        # Cache for inverse sqrt of C
        C_inv_sqrt = np.eye(dim)
        
        while calls < budget:
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                
                fy = func(y)
                fs.append(fy)
                calls += 1
                
                if fy < fbest:
                    fbest = fy
                    best_x = y.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            fs = np.array(fs)
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = fs[order]
            
            # Update mean
            x_new = sum(w[i] * ys[i] for i in range(mu))
            
            # Recompute C_inv_sqrt only when needed (every iteration is fine for dim=5)
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T
            
            # Update sigma with proper h_sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma
            
            # h_sigma: check if ps is consistent with expected norm
            E_norm_ps = np.sqrt(dim * (1 - (1 - c_sigma)**2))
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < 1.4 * E_norm_ps else 0.0
            
            sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / E_norm_ps - 1) * h_sigma)
            
            # Update covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))
            
            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (pc @ pc + (1 - (1 - c_c)**2) * C) \
                + c_mu * rankmu
            
            C = (C + C.T) / 2
            
            x0 = x_new
            B = sqrtm(C)
            
            # Check for convergence: sigma too small OR no progress in last 30 calls
            if sigma < 1e-7:
                break
            if calls > 30 and sigma < 1e-5:
                break
        
        return x0, sigma, best_x, fbest, calls
    
    def _polish(self, func, x_init, f_init, dim, remaining_budget):
        """Nelder-Mead polish on the best point found."""
        if remaining_budget < 20:
            return x_init, f_init
        
        # Nelder-Mead with max 200 iterations or until budget exhausted
        from scipy.optimize import minimize
        
        best_x = x_init.copy()
        best_f = f_init
        calls_made = 0
        max_calls = min(remaining_budget, 200)
        
        # Use a simple local search with decreasing step sizes
        step = 0.5
        for _ in range(max_calls):
            improved = False
            for d in range(dim):
                for sign in [1, -1]:
                    x_trial = best_x.copy()
                    x_trial[d] += sign * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial
                        improved = True
                    if calls_made >= max_calls:
                        break
                if calls_made >= max_calls:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
            elif calls_made >= max_calls:
                break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
