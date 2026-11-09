import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        dim = self.dim
        budget = self.budget
        
        # Storage for evaluated points
        self.X_evals = []
        self.Y_evals = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            # Clip to bounds just in case, though func does it
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            self.X_evals.append(x.copy())
            self.Y_evals.append(fx)
            return fx

        # Phase 1: Initial Exploration
        # Use a small number of points to find a good starting basin
        # Mix of random and origin-centered
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Generate initial points
        init_pts = []
        # 5 points near origin
        for _ in range(5):
            init_pts.append(rng.uniform(-1.0, 1.0, size=dim))
        # 15 points random in box
        for _ in range(15):
            init_pts.append(rng.uniform(-5.0, 5.0, size=dim))
        
        for x in init_pts:
            if self.call_count >= budget or time.time() > deadline - 10:
                break
            wrapped_func(x)
            
        if self.best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Multi-Start Local Optimization
        # Use top K initial points to start local optimizers
        if len(self.X_evals) > 0:
            # Sort by fitness
            indices = np.argsort(self.Y_evals)
            top_k = min(5, len(indices))
            top_indices = indices[:top_k]
            
            remaining_budget = budget - self.call_count
            if remaining_budget > 0:
                # Allocate budget to each start
                # Powell is robust and efficient
                budget_per_start = max(10, remaining_budget // top_k)
                
                for idx in top_indices:
                    if self.call_count >= budget or time.time() > deadline - 10:
                        break
                    
                    start_x = self.X_evals[idx].copy()
                    
                    def local_func(x):
                        if self.call_count >= budget or time.time() > deadline - 5:
                            raise ValueError("Budget or time exceeded")
                        return wrapped_func(x)
                    
                    try:
                        # Use Powell's method which is derivative-free and good for BBOB
                        res = minimize(
                            local_func,
                            start_x,
                            method='Powell',
                            options={
                                'maxiter': budget_per_start // 2,
                                'ftol': 1e-12,
                                'xtol': 1e-12
                            }
                        )
                    except Exception:
                        pass

        return self.best_x, self.best_f
