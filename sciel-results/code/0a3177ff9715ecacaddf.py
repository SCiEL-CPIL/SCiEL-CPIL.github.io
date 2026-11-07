import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

class Optimizer:
    """
    Hybrid Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Phase 1: Global Exploration using a small set of diverse starting points 
      (Origin, Sobol-like structured points, and random).
      We evaluate these points directly to establish a baseline and identify 
      promising basins.
    - Phase 2: Local Refinement using L-BFGS-B from the top K best starting points.
      L-BFGS-B is highly efficient for smooth functions like Rosenbrock, using
      quasi-Newton approximations. We strictly manage the call budget.
    - Phase 3: Final Polish. If budget remains, perform a few more L-BFGS-B steps
      or simple local perturbations around the global best.
    
    Key Improvements over pure Multi-Start:
    - Uses a smarter selection of starting points (not just uniform random).
    - Allocates budget dynamically: starts with exploration, then refines the 
      most promising candidates found during exploration.
    - Robust error handling and time management.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # Safety margin for 300s limit
        
        n = self.dim
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
                
            x_clipped = np.clip(x, -5, 5)
            calls_made += 1
            try:
                val = func(x_clipped)
            except Exception:
                return 1e10
            
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val
        
        # --- Phase 1: Global Exploration ---
        # Generate a diverse set of starting points.
        # We want to cover the space [-5, 5]^5 reasonably well.
        # 1. Origin
        # 2. Sobol-like points (using a simple low-discrepancy sequence or just stratified random)
        # 3. Random points
        
        n_starts = 12  # Total number of starting points to evaluate initially
        
        starts = []
        
        # 1. Origin
        starts.append(np.zeros(n))
        
        # 2. Stratified/Sobol-like points
        # For simplicity and robustness, we use a Halton sequence approximation or just stratified sampling.
        # Let's use a simple Halton sequence generator for low discrepancy.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate n_starts - 1 points using Halton sequence
        for i in range(1, n_starts):
            point = []
            for d in range(n):
                # Use distinct prime bases for each dimension
                base = [2, 3, 5, 7, 11][d % 5]
                # Map [0,1) to [-5, 5]
                val = -5 + 10 * halton(i, base)
                point.append(val)
            starts.append(np.array(point))
        
        # Evaluate all starting points
        # We must ensure we have enough budget for this.
        # n_starts = 12 calls. Budget = 500. Safe.
        
        candidates = []
        for x0 in starts:
            if calls_made >= self.budget or time.time() > deadline:
                break
            f0 = tracked_func(x0)
            candidates.append((f0, x0))
            
        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement ---
        # Take the top K candidates and run L-BFGS-B from each.
        # K should be small to allow deep refinement.
        # Let's pick the top 3-5 candidates.
        n_refine = 4
        top_candidates = candidates[:n_refine]
        
        # Allocate remaining budget among these candidates.
        # L-BFGS-B typically converges in 20-50 iterations for 5D Rosenbrock.
        # Each iteration costs ~dim+1 = 6 calls.
        # So 50 iterations ~ 300 calls.
        # We have ~488 calls left.
        # If we run 4 refinements, we have ~120 calls each.
        # That's ~20 iterations each. Might be enough.
        
        for f_init, x_init in top_candidates:
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Set a maxiter that fits the remaining budget roughly
            # but let the budget tracker be the hard stop.
            # We'll use a generous maxiter and rely on the tracker.
            
            try:
                # Use L-BFGS-B
                # ftol and gtol can be tight to squeeze out more improvement
                res = minimize(
                    tracked_func, 
                    x_init, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                
                # tracked_func already updates best_x and best_f
                # Check if we ran out of budget
                if calls_made >= self.budget:
                    break
                    
            except Exception:
                continue
        
        # --- Phase 3: Final Polish (Optional) ---
        # If we still have budget, try a very tight local search around the best point.
        # Or just let it be. L-BFGS-B usually does a good job.
        
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(n)
            if calls_made < self.budget:
                best_f = tracked_func(best_x)

        return best_x, best_f
