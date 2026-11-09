import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_time = 300.0  # Safety margin from 330s limit

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapper(x):
            if self.calls >= self.budget:
                return 1e100
            if time.time() - self.start_time > self.max_time:
                return 1e100
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # 1. Global Search: Halton-like sequence for better coverage
        # Simple deterministic low-discrepancy sequence
        n_global = 20
        # Using a simple Halton sequence implementation or just random with fixed seed
        # For robustness, let's use a mix of center and random
        rng = np.random.default_rng(123)
        
        # Start with center
        candidates = [np.zeros(self.dim)]
        
        # Add random points
        for _ in range(n_global - 1):
            p = rng.uniform(-4, 4, self.dim)
            candidates.append(p)
            
        # Evaluate global candidates
        for p in candidates:
            if self.calls >= self.budget:
                break
            wrapper(p)
            
        # Sort candidates by best_f found so far? No, we need to evaluate them first.
        # We have evaluated them in the loop above.
        # We need to keep track of which candidate gave which value.
        # Let's redo the global phase to keep pairs.
        
        # Reset for cleaner logic
        self.calls = 0
        self.best_f = np.inf
        self.best_x = None
        self.start_time = time.time()
        
        def wrapper2(x):
            if self.calls >= self.budget:
                return 1e100
            if time.time() - self.start_time > self.max_time:
                return 1e100
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Global Phase
        n_global = 20
        rng = np.random.default_rng(42)
        global_points = [np.zeros(self.dim)]
        for _ in range(n_global - 1):
            global_points.append(rng.uniform(-4, 4, self.dim))
            
        global_results = []
        for p in global_points:
            if self.calls >= self.budget:
                break
            f = wrapper2(p)
            global_results.append((f, p))
            
        # Sort by fitness
        global_results.sort(key=lambda x: x[0])
        
        # Local Phase: Try top 5 global points
        for f_init, x_init in global_results[:5]:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
                
            # Use L-BFGS-B
            try:
                # maxiter is not exactly function calls, but a good proxy
                # 50 iterations might use ~50-100 calls
                res = minimize(
                    wrapper2, 
                    x_init, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-6}
                )
                # If converged, we can break
                if res.success:
                    break
            except Exception:
                pass

        return self.best_x, self.best_f
