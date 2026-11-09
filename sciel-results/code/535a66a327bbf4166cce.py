"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """LHS-based portfolio optimizer with CMA-ES and L-BFGS-B polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_x = x.copy()
                best_f = f
        
        # Phase 1: Latin Hypercube Sampling for early progress
        # Use a good fraction of the budget for LHS to get a strong initial best
        lhs_calls = min(50, budget // 10)
        if lhs_calls > 0:
            # Generate LHS points in [-5, 5]^n
            # For dim=5, we can generate points efficiently
            # LHS: divide each dimension into lhs_calls strata
            # For each stratum, randomly permute the assignments
            points = np.zeros((lhs_calls, n))
            for d in range(n):
                # Create strata: [0, 1/lhs), [1/lhs, 2/lhs), ...
                # Random point in each stratum
                strata = np.arange(lhs_calls) / lhs_calls
                # Random offset within each stratum
                offsets = rng.random(lhs_calls) / lhs_calls
                points[:, d] = strata + offsets
                # Randomly permute to ensure full dimensionality coverage
                perm = rng.permutation(lhs_calls)
                points[:, d] = points[perm, d]
            
            # Map from [0,1] to [-5, 5]
            points = points * 10 - 5
            points = np.clip(points, -5, 5)
            
            # Evaluate LHS points
            for xi in points:
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break
                fi = func(xi)
                calls_made += 1
                update_best(xi, fi)
        
        # Phase 2: CMA-ES from the best LHS point
        # Use remaining budget for CMA-ES
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None:
            # Initialize CMA-ES around the best point
            x = best_x.copy()
            sigma = 1.0  # Start with moderate step size
            C = np.eye(n)
            Cs = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # CMA-ES parameters
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            cmu = 2 / ((n + 4)**2 + lam)
            ccs = (4 + lam / n) / (n + 4 + 2 * lam / n)
            cc = (2 + 2 * min(1.0, 1 - ccs) + cmu) / (n + 2 + 2 * min(1.0, 1 - ccs))
            cs = (n + 1.3) / (n + 5.5)
            d_sigma = (n + 2) / 2.0 + min(cmu, 0.5) * (mu_eff / 2.0 - 1.0)
            e_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
            
            while calls_made + lam <= budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate population
                Z = rng.standard_normal((lam, n))
                Y = Z @ Cs.T
                Pop = x + sigma * Y
                Pop = np.clip(Pop, -5, 5)
                
                # Evaluate population
                F = np.array([func(xi) for xi in Pop])
                calls_made += lam
                
                # Update best
                for xi, fi in zip(Pop, F):
                    update_best(xi, fi)
                
                # Sort by fitness
                idx = np.argsort(F)
                idx_top = idx[:mu]
                
                # Update mean
                x_new = np.sum(w[:, None] * Pop[idx_top], axis=0)
                
                # Update evolution path ps
                diff = x_new - x
                try:
                    y = np.linalg.solve(Cs.T, diff)
                except np.linalg.LinAlgError:
                    y = diff / (np.linalg.norm(diff) + 1e-10)
                
                ps = (1 - cs) * ps + cs / sigma * np.sqrt(mu_eff) * y
                
                # Update pc
                thresh = (1.4 + 2 / (n + 1)) * e_norm
                hsig = 1.0 if np.linalg.norm(ps) < thresh else 0.0
                pc = (1 - cc) * pc + hsig * cc * (x_new - x) / sigma
                
                # Update C
                Y_top = (Pop[idx_top] - x[None, :]) / sigma
                C = (1 + cmu * (mu_eff + 2) - cc) * C + cmu * (Y_top.T * w) @ Y_top
                
                # Update sigma
                sigma = sigma * np.exp(cs / d_sigma * (np.linalg.norm(ps) / e_norm - 1.0))
                sigma = np.clip(sigma, 1e-8, 10)
                
                # Update Cs (inverse square root)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    Cs = eigvecs @ np.diag(np.sqrt(eigvals))
                except np.linalg.LinAlgError:
                    Cs = np.eye(n)
                    C = np.eye(n)
                
                # Update x
                x = x_new
        
        # Phase 3: L-BFGS-B polishing if budget remains
        # Use scipy.optimize.minimize with L-BFGS-B
        if calls_made < budget - 5 and best_x is not None:
            try:
                from scipy.optimize import minimize
                # Wrap func to count calls
                polish_calls = [0]
                max_polish_calls = budget - calls_made - 1
                
                def wrapped_func(x):
                    nonlocal calls_made, best_x, best_f
                    if polish_calls[0] >= max_polish_calls or time.time() > deadline - 3:
                        return best_f  # Return current best to stop
                    fx = func(x)
                    calls_made += 1
                    polish_calls[0] += 1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                    return fx
                
                res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n, options={'maxiter': 100, 'maxfun': max_polish_calls})
            except Exception:
                pass
        
        # Return the best solution found
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1
        
        return best_x, best_f

# EVOLVE-BLOCK-END
