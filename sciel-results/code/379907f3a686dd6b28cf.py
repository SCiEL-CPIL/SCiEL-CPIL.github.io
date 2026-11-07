"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        best_x = np.full(self.dim, 0.0)
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        
        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            if len(x) != self.dim:
                x = x[:self.dim] if len(x) > self.dim else np.pad(x, (0, self.dim - len(x)))
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))
        
        # Phase 1: CMA-ES from a few random starts
        # We'll do multiple short runs of CMA-ES-like evolution
        
        # Use a simple CMA-ES implementation for robustness
        dim = self.dim
        
        def cma_es_run(x0, f0, sigma0, max_iter, time_limit):
            nonlocal best_x, best_f, calls_made, budget
            x = x0.copy()
            f = f0
            sigma = sigma0
            dim = self.dim
            mu = max(1, dim // 2)
            lambda_ = 4 * dim
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            h_sigma = (1.4 + 2 / (dim + 1))
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            for i in range(max_iter):
                if time.time() > time_limit:
                    break
                if calls_made >= budget:
                    break
                
                # Generate offspring
                ys = []
                for k in range(lambda_):
                    y = x + sigma * (B @ (D * rng.standard_normal(dim)))
                    ys.append(y)
                
                fs = []
                for y in ys:
                    fy = safe_func(y)
                    if fy < f:
                        f = fy
                        x = y.copy()
                        if f < best_f:
                            best_f = f
                            best_x = y.copy()
                    fs.append(fy)
                
                # Sort by fitness
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Update mean
                y_w = np.array(ys[:mu])
                x_new = y_w.T @ weights
                x = x_new
                
                # Update step size
                y_sigma = (B @ (D * (y_w.T @ weights))) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_sigma
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (i + 1))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance matrix
                delta_h = int(np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (i + 1))))
                pc = (1 - c_c) * pc + c_c * delta_h * (1 - c_sigma) / sigma * y_sigma
                rank_one = np.outer(pc, pc)
                rank_mu = (1 / (sigma ** 2)) * np.sum(weights[:k] * np.outer(ys[k] - x, ys[k] - x) for k in range(mu))
                C = (1 + c_c * (2 - delta_h - c_c) * rank_one - c_c * rank_mu) * C if False else None
                
                # Simplified: just use diagonal adaptation for speed
                # Actually, let's use a simpler strategy: adapt sigma and use a random direction restart
                break  # Simplified: just do local search
            
            return x, f
        
        # Better approach: Use scipy's Nelder-Mead and Powell with multiple restarts
        # First, do a coarse grid search to find promising regions
        n_grid = 3
        grid_points = np.meshgrid(*[np.linspace(-5, 5, n_grid) for _ in range(dim)])
        grid_points = np.array(grid_points).reshape(dim, -1).T
        
        # Evaluate a subset of grid points
        evaluated = []
        for gp in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f_gp = safe_func(gp)
            evaluated.append((f_gp, gp))
        
        if evaluated:
            evaluated.sort(key=lambda x: x[0])
            # Take top 3 starting points
            top_starts = [gp for _, gp in evaluated[:3]]
        else:
            top_starts = [np.zeros(dim)]
        
        # Phase 2: Local optimization from each starting point
        for x0 in top_starts:
            if calls_made >= budget or time.time() > deadline:
                break
            
            try:
                result = scipy_minimize(safe_func, x0, method='Nelder-Mead', 
                                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            if calls_made >= budget or time.time() > deadline:
                break
            
            try:
                result = scipy_minimize(safe_func, x0, method='Powell',
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final polish with CMA-ES-like local search if budget remains
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x = best_x.copy()
            f = best_f
            for _ in range(min(100, budget - calls_made)):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < f:
                    x, f = y, fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
