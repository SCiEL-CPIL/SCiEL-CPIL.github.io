import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        deadline = self.start_time + 295.0 # 5s margin
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Global exploration with 50 uniform random evaluations
        n_global = 50
        best_x = None
        best_f = np.inf
        
        for _ in range(n_global):
            if time.time() > deadline or self.call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            self.call_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            x = rng.uniform(-5, 5, self.dim)
            self.call_count += 1
            best_x = x
            best_f = func(x)
        
        # Phase 2: CMA-ES for local optimization
        # Use scipy.optimize.cmaes if available, otherwise fallback to a simple CMA-ES implementation
        try:
            from cmaes import CMA
            import cmaes
        except ImportError:
            CMA = None
            cmaes = None
        
        remaining_budget = self.budget - self.call_count
        max_iter = max(1, remaining_budget // 10)  # Approximate iterations needed
        
        if CMA is not None:
            # Configure CMA-ES
            opts = {
                'maxfevals': remaining_budget,
                'seed': 0,
                'verbose': -1,
                'bounds': [(-5.0, 5.0)] * self.dim
            }
            
            # Initialize CMA-ES from best point
            cma = CMA(best_x, 1.0, opts)
            
            while not cma.should_stop():
                if time.time() > deadline or self.call_count >= self.budget:
                    break
                
                # Generate a population
                pop = cma.ask()
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if self.call_count >= self.budget or time.time() > deadline:
                        # Fill remaining with best value to avoid breaking CMA-ES
                        fvals.append(best_f)
                    else:
                        x_clipped = np.clip(x, -5, 5)
                        self.call_count += 1
                        fx = func(x_clipped)
                        fvals.append(fx)
                
                cma.tell(pop, fvals)
                
                # Update best
                if cma.result.xbest is not None:
                    best_x = np.clip(cma.result.xbest, -5, 5)
                    best_f = cma.result.fbest
        
        else:
            # Fallback: Simple (1+1) ES with adaptive step size, but better tuned
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            
            for _ in range(remaining_budget):
                if time.time() > deadline or self.call_count >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                self.call_count += 1
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.95
            best_x = x
            best_f = fx
        
        return best_x, best_f
