import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x[:n]
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val
        
        # Phase 1: Global exploration with multiple starting points
        # Use Latin Hypercube Sampling for diverse coverage
        num_starts = 20
        lhs_points = []
        for i in range(num_starts):
            # Latin Hypercube: divide each dimension into num_starts segments
            perm = rng.permutation(num_starts)
            point = np.zeros(n)
            for d in range(n):
                seg = perm[d]
                # Uniform within segment
                u = rng.uniform(seg / num_starts, (seg + 1) / num_starts)
                point[d] = -5 + 10 * u
            lhs_points.append(point)
        
        # Add center and a few random points
        lhs_points.append(np.zeros(n))
        for _ in range(5):
            lhs_points.append(rng.uniform(-5, 5, size=n))
        
        # Evaluate all starting points
        for p in lhs_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            p = np.clip(p, -5, 5)
            fp = safe_func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # Phase 2: Local refinement using Nelder-Mead from top candidates
        # We'll run Nelder-Mead from the best few starting points
        # Allocate remaining budget across multiple local searches
        
        # Collect top 5 starting points based on function value
        # We already have best_x, but let's try from multiple good points
        # For efficiency, we'll do local search from best_x first, then others if budget remains
        
        # Nelder-Mead options
        nm_options = {
            'maxiter': 100,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        # Try local search from best point
        if calls_made < budget and time.time() < deadline:
            try:
                res = minimize(safe_func, best_x.copy(), method='Nelder-Mead', options=nm_options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Try local search from other good points if budget remains
        # We need to track which points were good
        # For simplicity, let's do a few more local searches from random points near best
        if calls_made < budget and time.time() < deadline:
            # Generate a few points near the best
            for attempt in range(5):
                if calls_made >= budget or time.time() >= deadline:
                    break
                # Perturb best_x
                sigma = 1.0
                x_init = best_x + sigma * rng.standard_normal(n)
                x_init = np.clip(x_init, -5, 5)
                
                # Quick evaluation
                fx = safe_func(x_init)
                if fx < best_f:
                    best_f = fx
                    best_x = x_init.copy()
                
                # Local search from this point
                if calls_made < budget and time.time() < deadline:
                    try:
                        res = minimize(safe_func, x_init.copy(), method='Nelder-Mead', options=nm_options)
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass
        
        # Phase 3: (1+8)-ES refinement for the sharp ridge
        # This is inspired by the finding that (1+8)-ES works better than (1+1)-ES
        if calls_made < budget and time.time() < deadline:
            sigma_11 = 0.5
            lam_es = 8
            while calls_made + lam_es <= budget and time.time() < deadline:
                # Generate lam_es candidates
                candidates = []
                for _ in range(lam_es):
                    z = rng.standard_normal(n)
                    y = best_x + sigma_11 * z
                    y = np.clip(y, -5, 5)
                    candidates.append(y)
                
                # Evaluate all candidates
                fs = np.array([safe_func(y) for y in candidates])
                
                # Find best candidate
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = candidates[idx_best].copy()
                    # Increase sigma on success
                    sigma_11 *= 1.1
                else:
                    # Decrease sigma on failure
                    sigma_11 *= 0.85
                
                sigma_11 = np.clip(sigma_11, 1e-8, 10.0)
                
                # Check convergence
                if sigma_11 < 1e-6:
                    break
        
        # Phase 4: Final fine-tuning with small steps
        if calls_made < budget and time.time() < deadline:
            sigma_fine = 0.1
            while calls_made < budget and time.time() < deadline:
                z = rng.standard_normal(n)
                y = best_x + sigma_fine * z
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_fine *= 1.05
                else:
                    sigma_fine *= 0.95
                
                sigma_fine = np.clip(sigma_fine, 1e-8, 10.0)
                if sigma_fine < 1e-7:
                    break
        
        return best_x, best_f
