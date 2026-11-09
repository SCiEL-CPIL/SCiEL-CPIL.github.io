import numpy as np
import time
import math

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        rng = np.random.default_rng(42)
        
        def try_eval(x):
            nonlocal best_f, best_x, eval_count
            if eval_count >= self.budget or time.time() > deadline - 2:
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for better initial coverage
        n_init = 20
        # Generate LHS points in [0, 1]^dim
        indices = [rng.permutation(n_init) for _ in range(self.dim)]
        lhs_points = np.array([
            [(idx[i] + rng.uniform()) / n_init for idx in indices] 
            for i in range(n_init)
        ])
        # Scale to [-5, 5]
        lhs_points = lhs_points * 10 - 5
        
        for x in lhs_points:
            try_eval(x)
        
        # Phase 2: CMA-ES for global convergence
        if HAS_CMA:
            try:
                x0 = best_x.tolist()
                # Larger sigma to ensure global exploration initially
                sigma0 = 2.0 
                popsize = 10 # Slightly larger population for better exploration
                
                remaining_budget = self.budget - eval_count
                # Use 80% of remaining for CMA
                cma_budget = int(remaining_budget * 0.8)
                maxiter = max(1, (cma_budget - 1) // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 1,
                    'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                    'CMA_adaptation_of_maxiter': True,
                    'CMA_adaptation_of_maxfevals': True,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget:
                    if time.time() > deadline - 15:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget:
                            break
                        if time.time() > deadline - 15:
                            break
                        fx = func(x)
                        eval_count += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass

        # Phase 3: Local Polish with (1+1)-ES
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(123)
            x = best_x.copy()
            fx = best_f
            
            # Start with a small sigma for refinement
            sigma = 0.1
            no_improve = 0
            limit = 20
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.05
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-5:
                        break
                        
        return best_x, best_f
