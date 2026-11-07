"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf
        self.history = []
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, keep 10s safety margin
        self.deadline = self.start_time + 290.0
        
        def time_ok():
            return time.time() < self.deadline

        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            self.history.append((fx, x.copy()))
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search with Sobol Sequence
        # We want a good spread. 100 points is a reasonable balance for 5D.
        n_global = 100
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_global)
        except ImportError:
            rng = np.random.default_rng(42)
            sobol_points = rng.random((n_global, self.dim))
        
        # Map [0, 1] to [-5, 5]
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Add origin and some random points to ensure diversity?
        # Sobol is already good. Let's just use Sobol.
        
        for x0 in sobol_points:
            if not time_ok() or self.call_count >= self.budget - 10:
                break
            try:
                tracked_func(x0)
            except Exception:
                break

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Sort history by fx
        valid_history = [(fx, x) for fx, x in self.history if np.isfinite(fx)]
        valid_history.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization from Top Candidates
        remaining_budget = self.budget - self.call_count
        
        # We want to use the remaining budget effectively.
        # Let's pick the top K candidates.
        # If we have 400 budget left, and we pick 5 candidates, we can give 80 evals each.
        # If we pick 10 candidates, we give 40 evals each.
        # Rosenbrock needs decent steps. 40 is tight. 80 is better.
        
        max_starts = 5
        # Ensure we don't exceed budget
        if remaining_budget > 50:
             # Allocate budget to starts
             # Leave 10 for final polish
             usable_budget = remaining_budget - 10
             # Try to use up to 5 starts
             n_starts = min(max_starts, len(valid_history))
             # If n_starts is 0, something is wrong
             if n_starts == 0:
                 n_starts = 1
             
             # Calculate evals per start
             evals_per_start = usable_budget // n_starts
             # Cap evals per start to avoid one start eating all budget
             # But for Rosenbrock, more is better if in basin
             evals_per_start = min(evals_per_start, 100)
             
             for i in range(n_starts):
                 if not time_ok() or self.call_count >= self.budget - 5:
                     break
                 
                 if i < len(valid_history):
                     _, x_start = valid_history[i]
                 else:
                     break
                     
                 try:
                     bounds = [(-5.0, 5.0)] * self.dim
                     res = minimize(
                         tracked_func,
                         x_start,
                         method='L-BFGS-B',
                         bounds=bounds,
                         options={
                             'maxiter': 200, 
                             'maxfun': evals_per_start, 
                             'ftol': 1e-12, 
                             'gtol': 1e-10
                         }
                     )
                 except Exception:
                     pass

        # Phase 3: Final Polish
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 5 and time_ok() and self.best_x is not None:
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                # Use remaining budget for a high-precision polish
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfun': remaining_budget - 2, 
                        'ftol': 1e-15, 
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
