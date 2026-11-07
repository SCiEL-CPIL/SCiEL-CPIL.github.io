"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global-local optimizer: multi-start Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        
        # Track number of function evaluations
        eval_count = 0
        
        def count_func(x):
            nonlocal eval_count
            eval_count += 1
            return func(x)
        
        # Initial multi-start with diverse points
        # Use a mix of random points and structured points
        n_starts = 8
        starts = []
        
        # Add some structured starting points
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 2.5)
        starts.append(-np.ones(self.dim) * 2.5)
        
        # Add random points
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # For each start, run Nelder-Mead with a limited budget
        # Total budget is 500, so we need to be careful
        # Let's allocate budget adaptively: start with small budgets, increase if promising
        
        # First, evaluate all starting points
        for s in starts:
            if eval_count >= self.budget:
                break
            fs = count_func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
        
        # Now run local optimization from the best few starts
        # Sort starts by their function values
        start_results = []
        for s in starts:
            if eval_count >= self.budget:
                break
            # Re-evaluate if not already done (simple approach: just track)
            # Actually, let's just run Nelder-Mead from the best start first
            break
        
        # Run Nelder-Mead from the best starting point
        if best_x is not None and eval_count < self.budget:
            # Use Nelder-Mead with careful parameter settings
            # The function is Rastrigin-like (rotated), so it's multimodal
            # Nelder-Mead may get stuck in local minima, but with good starting points it can help
            
            # Set up options for Nelder-Mead
            # maxiter is the number of iterations, not function evaluations
            # Each iteration uses dim+1 function evaluations roughly
            # So maxiter should be (budget - eval_count) / (dim + 1)
            remaining = self.budget - eval_count
            max_iter = max(1, int(remaining / (self.dim + 1)))
            
            try:
                result = minimize(
                    count_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # If we have budget left, do more multi-start local optimization
        while eval_count < self.budget - (self.dim + 1):
            # Generate a new random starting point
            s = rng.uniform(-4, 4, self.dim)
            fs = count_func(s)
            
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
            
            # Run a short Nelder-Mead from this point
            remaining = self.budget - eval_count
            if remaining < (self.dim + 2):
                break
            
            max_iter = max(1, int(remaining / (self.dim + 1)) // 2)
            
            try:
                result = minimize(
                    count_func,
                    s,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-3,
                        'fatol': 1e-3,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
