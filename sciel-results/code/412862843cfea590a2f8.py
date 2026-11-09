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
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        # Phase 1: Initial random sampling
        n_init = min(30, self.budget // 5)
        rng = np.random.default_rng(0)
        
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES for global convergence
        if HAS_CMA:
            try:
                x0 = best_x.tolist()
                sigma0 = 0.5
                popsize = 4 + int(3 * math.log(self.dim))
                if popsize < 4:
                    popsize = 4
                
                remaining_budget = self.budget - eval_count
                # Use 70% of remaining for CMA, leave 30% for BO + local polish
                cma_budget = int(remaining_budget * 0.7)
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
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget:
                    if time.time() > deadline - 10:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget:
                            break
                        if time.time() > deadline - 10:
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

        # Phase 3: Nelder-Mead for refinement
        if HAS_SCIPY_OPT and eval_count < self.budget and time.time() < deadline - 10:
            try:
                # Wrapper to track evaluations and enforce deadline
                def wrapper(x):
                    nonlocal eval_count, best_f, best_x
                    if eval_count >= self.budget or time.time() > deadline - 5:
                        # Return a large value to discourage further exploration?
                        # Or just return best_f? 
                        # Minimizing a constant might cause issues.
                        # Let's just return the function value if within budget.
                        # The check is primarily to stop the loop in minimize if we hit the limit.
                        # However, minimize doesn't easily stop on callback.
                        # We rely on maxiter and the time check in the outer scope is not possible inside minimize.
                        # So we must ensure maxiter is conservative enough.
                        pass
                    
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    eval_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                # Estimate max iterations for Nelder-Mead
                nm_budget = self.budget - eval_count
                # Nelder-Mead uses dim+1 function evaluations per iteration roughly
                maxiter_nm = max(10, nm_budget // (self.dim + 1))
                
                res = minimize(wrapper, best_x, method='Nelder-Mead', 
                               options={'maxiter': maxiter_nm, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 4: Local Polish: (1+1)-ES with adaptive step size
        remaining = self.budget - eval_count
        if remaining < 0:
            remaining = 0
            
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.1
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-5:
                        break
                        
        return best_x, best_f


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
