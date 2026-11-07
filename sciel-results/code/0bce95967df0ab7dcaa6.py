"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

def generate_sobol_sequence(n, dim, seed=42):
    """
    Generate a Sobol sequence in [0, 1]^dim.
    Since scipy.stats.qmc is available in recent versions, we use it.
    If not, we fall back to Halton or random.
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        points = sampler.random(n)
        return points
    except ImportError:
        # Fallback to Halton-like or just random if qmc is not available
        # For simplicity and robustness, we use random if qmc fails, 
        # but qmc is standard in modern scipy.
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))

class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) in 5D.
    
    Strategy:
    1. Global search using Sobol sequence to find a promising basin.
    2. Local optimization using L-BFGS-B from the top candidates.
    3. Polishing with L-BFGS-B if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf
        self.history = [] # List of (fx, x)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Wrapper to track calls and enforce budget/deadline
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

        # Phase 1: Global Search using Sobol Sequence
        # We want to cover the space [-5, 5]^dim.
        # Sobol points are in [0, 1]^dim. Map to [-5, 5].
        
        # How many points for global search?
        # 50 points is a good balance. 2^5 = 32 corners + some.
        # Let's use 40 Sobol points + origin.
        n_global = 40
        sobol_points = generate_sobol_sequence(n_global, self.dim, seed=42)
        # Map [0, 1] to [-5, 5]
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Add origin
        sobol_points = np.vstack([np.zeros(self.dim), sobol_points])
        
        # Evaluate global points
        for x0 in sobol_points:
            if not time_ok() or self.call_count >= self.budget - 20:
                break
            try:
                tracked_func(x0)
            except Exception:
                pass

        if self.best_x is None:
             return np.zeros(self.dim), 0.0

        # Sort history by fx to get top candidates
        # Only consider finite values
        valid_history = [(fx, x) for fx, x in self.history if np.isfinite(fx)]
        valid_history.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization from Top Candidates
        # We will run L-BFGS-B from the top K candidates.
        # K depends on remaining budget.
        remaining_budget = self.budget - self.call_count
        
        if remaining_budget > 10 and time_ok():
            # Determine number of starts
            # Each L-BFGS-B run might use up to 50-100 evals.
            # Let's allow up to 3-5 starts if budget permits.
            max_starts = 5
            # But we need to reserve budget.
            # If we have 100 budget left, 2 starts of 50 is safe.
            # If we have 200 budget left, 4 starts of 50 is safe.
            
            # Heuristic: 30 evals per start is a good cap for L-BFGS-B in 5D Rosenbrock.
            evals_per_start = 30
            n_starts = min(max_starts, len(valid_history), max(1, (remaining_budget - 10) // evals_per_start))
            
            for i in range(n_starts):
                if not time_ok() or self.call_count >= self.budget - 5:
                    break
                
                # Get the i-th best candidate
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
                            'maxiter': 50, 
                            'maxfun': evals_per_start, 
                            'ftol': 1e-10, 
                            'gtol': 1e-8
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Polishing
        # If we have budget left, run L-BFGS-B again from the current best with tighter tolerances.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 5 and time_ok() and self.best_x is not None:
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining_budget - 2, 
                        'ftol': 1e-15, 
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        # Ensure we return a valid solution
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
