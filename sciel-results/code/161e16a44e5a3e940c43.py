"""HYBRID: Rastrigin-aware grid basin identification with DE refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Hybrid optimizer: coarse grid to find Rastrigin basins, then DE refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit (evaluator kills at 330s)

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            
            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        def evaluate(x):
            """Evaluate a point, updating global best."""
            return counted_func(x)

        # Phase 1: Coarse grid search to identify promising basins
        # Büche-Rastrigin has local minima near integer points
        # Use a grid with spacing ~2 to cover [-5,5]^5 efficiently
        grid_points = []
        # Generate grid: 4 points per dimension (-4, -2, 0, 2, 4) = 5^5 = 3125
        # Too many. Use 3 points per dim: (-3, 0, 3) => 3^5 = 243
        # Plus add integer points in a subset
        grid_vals = [-4.0, -2.0, 0.0, 2.0, 4.0]
        
        # Sample a subset of the grid to stay within budget
        # Use stratified sampling: pick ~100 points from the full grid
        all_grid = np.array(np.meshgrid(*[grid_vals]*self.dim)).reshape(self.dim, -1).T
        # Shuffle and take first N
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_grid))
        num_grid = min(150, len(all_grid), max_calls // 4)
        
        for i in range(num_grid):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = all_grid[idx[i]]
            evaluate(x)
        
        # Phase 2: Local refinement around top grid points
        # Collect all evaluated points and their values
        # We'll do Nelder-Mead refinement around the best few grid points
        # Track evaluated points
        eval_points = []
        eval_values = []
        
        # Re-evaluate top candidates with local optimization
        # Sort by value (we need to track which points were evaluated)
        # Since we can't easily retrieve, do a second pass on top regions
        
        # Use the best point as anchor for local search
        if best_f < float('inf'):
            # Nelder-Mead around best
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
            except Exception:
                pass
        
        # Phase 3: DE refinement if budget remains
        if call_count < max_calls and time.time() < deadline:
            bounds = [(-5.0, 5.0)] * self.dim
            remaining = max_calls - call_count
            # Scale popsize and maxiter to use remaining budget
            # DE uses popsize * dim * maxiter evaluations roughly
            # popsize=10, maxiter=10 => ~500 evals, too many if remaining is small
            # Adjust: popsize=8, maxiter=5 => ~200 evals
            popsize = max(5, min(10, remaining // (self.dim * 3)))
            maxiter = max(3, min(15, remaining // (popsize * self.dim)))
            
            try:
                differential_evolution(
                    counted_func,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    seed=42,
                    init='latinhypercube'
                )
            except Exception:
                pass

        # Phase 4: Final local polish if time allows
        if call_count < max_calls and time.time() < deadline:
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
