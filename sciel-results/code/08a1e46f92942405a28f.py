import numpy as np
import time

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
        deadline = t0 + 315  # Safety margin: 315s out of 330s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        
        # Helper: Evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling to establish a baseline
        # Use a mix of random points and points near origin
        n_init = 20
        for i in range(n_init):
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            if i < 5:
                # Points near origin
                x = rng.uniform(-1, 1, size=dim)
            else:
                # Random points in the box
                x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        # Phase 2: CMA-ES for global search
        # CMA-ES is strong for BBOB-style functions like Discus (f11)
        if HAS_CMA and eval_count < self.budget - 20:
            try:
                # Start CMA-ES from the best point found so far
                x0 = best_x.tolist()
                sigma0 = 1.0
                popsize = 10
                
                # Use 70% of remaining budget for CMA-ES
                remaining_budget = self.budget - eval_count
                cma_budget = int(remaining_budget * 0.7)
                maxiter = max(1, (cma_budget - 1) // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 42,
                    'bounds': [-5.0] * dim + [5.0] * dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    # Relax stopping criteria to ensure we use the allocated budget
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget - 20:
                    if time.time() > deadline - 10:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget - 20:
                            break
                        if time.time() > deadline - 10:
                            break
                        fx = evaluate(x)
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass
        
        # Phase 3: Local polish with (1+1)-ES
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            limit = 30
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    # Restart with larger sigma
                    sigma = 0.1
                    no_improve = 0
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-4:
                        break
        
        return best_x, best_f
