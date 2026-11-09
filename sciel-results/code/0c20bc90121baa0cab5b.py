"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            return float(func(x))
        
        # Phase 1: Differential Evolution (Global Search)
        N = 10
        F = 0.8
        CR = 0.9
        max_gen_de = 20
        
        for r in range(2):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            if r == 0:
                pop = np.zeros((N, self.dim))
                for i in range(N):
                    if i == 0:
                        pop[i] = 0.0
                    else:
                        pop[i] = rng.uniform(-5, 5, self.dim)
            else:
                pop = rng.uniform(-5, 5, (N, self.dim))
            
            pop_fit = np.array([safe_func(x) for x in pop])
            
            for gen in range(max_gen_de):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                for i in range(N):
                    if calls_used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    candidates = [j for j in range(N) if j != i]
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    u = np.zeros(self.dim)
                    j_rand = rng.integers(0, self.dim)
                    mask = rng.random(self.dim) < CR
                    mask[j_rand] = True
                    u[mask] = v[mask]
                    u[~mask] = pop[i][~mask]
                    
                    u = np.clip(u, -5.0, 5.0)
                    f_u = safe_func(u)
                    
                    if f_u <= pop_fit[i]:
                        pop[i] = u
                        pop_fit[i] = f_u
                        if f_u < best_f:
                            best_f = f_u
                            best_x = u.copy()
        
        # Phase 2: Nelder-Mead Local Refinement
        if best_x is not None and calls_used < self.budget and time.time() < deadline:
            def nm_func(x):
                return safe_func(x)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
