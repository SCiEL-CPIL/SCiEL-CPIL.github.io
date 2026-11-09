import numpy as np
import time
from scipy.optimize import minimize

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: 330s limit, use 320s to be safe
        deadline = t0 + 320.0
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        rng = np.random.default_rng(42)
        
        def check_deadline():
            return time.time() < deadline - 5.0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Diverse Sampling
        # Use a small fraction of the budget to explore the space and find a good basin.
        # This helps CMA-ES avoid starting in a bad local minimum.
        n_init = max(10, int(self.budget * 0.1))
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            # Mix of points near origin and random points
            if i < 5:
                x = rng.uniform(-1, 1, size=dim)
            else:
                x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        # Phase 2: CMA-ES for Global/Intermediate Search
        # CMA-ES is robust for the Discus landscape due to covariance adaptation.
        if HAS_CMA and eval_count < self.budget - 20 and check_deadline():
            try:
                x0 = best_x.tolist()
                sigma0 = 1.0
                popsize = 10
                
                # Allocate ~75% of remaining budget to CMA-ES
                remaining_budget = self.budget - eval_count
                cma_budget = int(remaining_budget * 0.75)
                maxiter = max(1, (cma_budget - 1) // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 42,
                    'bounds': [-5.0] * dim + [5.0] * dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    # Relax stopping criteria to use full allocated budget
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget - 20 and check_deadline():
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget - 20 or not check_deadline():
                            break
                        fx = evaluate(x)
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass

        # Phase 3: Nelder-Mead Local Polish
        # Use the remaining budget for a deterministic, gradient-free local search.
        # Nelder-Mead is effective for final convergence in low dimensions.
        remaining = self.budget - eval_count
        if remaining > 5 and check_deadline():
            # We will run Nelder-Mead from the current best point.
            # To be robust, we might run it a few times if budget allows, 
            # but a single run with a good maxiter is usually sufficient for 5D.
            
            # Define the function wrapper for scipy
            def nm_func(x):
                if eval_count >= self.budget or not check_deadline():
                    return best_f # Return best known to stop early effectively
                return evaluate(x)
            
            # Set maxiter to use up the remaining budget
            # Nelder-Mead uses roughly 1-2 evals per iteration in low dim
            maxiter_nm = max(1, int(remaining * 0.9))
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter_nm,
                        'adaptive': True,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'initial_simplex': None # Let scipy choose, or we can specify
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
