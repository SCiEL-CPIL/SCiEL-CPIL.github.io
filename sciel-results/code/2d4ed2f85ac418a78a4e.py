import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.func = None

    def _call_func(self, x):
        if self.calls >= self.budget:
            return float('inf')
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def _ternary_search(self, x, dim, lo, hi, tol=1e-8, max_iter=30):
        best_val = float('inf')
        best_pos = x[dim]
        
        for _ in range(max_iter):
            if self.calls >= self.budget - 1:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            
            x[dim] = m1
            f1 = self._call_func(x)
            x[dim] = m2
            f2 = self._call_func(x)
            
            if f1 < f2:
                hi = m2
                if f1 < best_val:
                    best_val = f1
                    best_pos = m1
            else:
                lo = m1
                if f2 < best_val:
                    best_val = f2
                    best_pos = m2
            
            if hi - lo < tol:
                break
                
        x[dim] = best_pos
        return best_pos

    def __call__(self, func):
        self.func = func
        rng = np.random.default_rng(42)
        
        # Initial random point
        x = rng.uniform(-5, 5, self.dim)
        self._call_func(x)
        
        start_time = time.time()
        deadline = start_time + 280
        
        dim_order = list(range(self.dim))
        rng.shuffle(dim_order)
        
        cycle = 0
        prev_best_f = self.best_f
        
        while self.calls < self.budget:
            if time.time() > deadline:
                break
            
            # Determine max_iter based on cycle
            # First cycle: 30, subsequent: 10
            max_iter = 30 if cycle == 0 else 10
            
            for d in dim_order:
                if self.calls >= self.budget - 1:
                    break
                self._ternary_search(x, d, -5.0, 5.0, tol=1e-8, max_iter=max_iter)
            
            cycle += 1
            
            # Early stopping if no significant improvement
            if self.best_f < prev_best_f - 1e-12:
                prev_best_f = self.best_f
            else:
                # If no improvement, we might be converged
                # But let's do one more check or break if budget is low
                if cycle > 1:
                    break

        return self.best_x, self.best_f
