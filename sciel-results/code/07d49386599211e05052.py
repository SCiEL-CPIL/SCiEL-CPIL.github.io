"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Local Search with Halton sequence initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf

    def _halton(self, index, base):
        """Generate the index-th Halton sequence value for a given base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i = i // base
        return r

    def _halton_sequence(self, n, dim):
        """Generate n points in dim dimensions using Halton sequences."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                # Use the (j+1)-th prime as base for dimension j
                base = 2
                for k in range(j + 1, j + 10):
                    is_prime = True
                    for p in range(2, int(np.sqrt(k)) + 1):
                        if k % p == 0:
                            is_prime = False
                            break
                    if is_prime:
                        base = k
                        break
                # Skip if base is not found (should not happen for small dims)
                if base == 2 and j > 0:
                     # Fallback to simple random if prime search fails (unlikely for dim=5)
                     points[i, j] = np.random.rand()
                else:
                    points[i, j] = self._halton(i + 1, base)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin of 40s before 330s limit
        
        # State variables
        self.best_x = None
        self.best_f = np.inf
        
        dim = self.dim
        budget = self.budget
        
        # Split budget among restarts
        # We use 10 restarts. Each gets 30 evaluations.
        n_restarts = 10
        budget_per_restart = max(1, budget // n_restarts)
        
        # Generate initial points using Halton sequence for uniform coverage
        init_points = self._halton_sequence(n_restarts, dim)
        
        # Map [0, 1]^dim to [-5, 5]^dim
        init_points = init_points * 10.0 - 5.0
        
        # We also add the center and a few random points to ensure diversity
        # But Halton is usually sufficient. Let's stick to Halton + maybe one center.
        # Actually, let's just use the Halton points.
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            
            x0 = init_points[i]
            
            # Define objective wrapper to count calls and check time
            calls = [0]
            def objective(x):
                calls[0] += 1
                if time.time() > deadline:
                    return np.inf
                return func(x)
            
            # Run Nelder-Mead local search
            # maxiter is limited by budget_per_restart
            # nfev limit is strict
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': budget_per_restart, 'maxfev': budget_per_restart, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                # If optimization fails, at least evaluate x0 if not done
                pass

            # Check if we have used up the total budget
            if calls[0] >= budget:
                break
        
        # If no valid point was found (shouldn't happen), return center
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_f = func(self.best_x)
            
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
