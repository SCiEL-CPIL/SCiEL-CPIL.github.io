"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using Multi-Start Nelder-Mead with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Solver using Multi-Start Nelder-Mead with Latin Hypercube Sampling initialization.
    
    Strategy:
    1. Use a portion of the budget for Latin Hypercube Sampling (LHS) to explore the search space
       and identify a promising region, handling the hidden shift and rotation.
    2. Run Nelder-Mead from the best point found in the sampling phase.
    3. Implement a restart strategy with decreasing simplex spread if stagnation is detected.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Exploration with Latin Hypercube Sampling (LHS)
        # Allocate 10% of budget to exploration, min 10, max 50
        n_samples = max(10, min(50, int(budget * 0.1)))
        
        # Generate LHS points
        rng = np.random.default_rng(42)
        
        # LHS implementation:
        # Divide each dimension into n_samples intervals, pick one random point in each interval.
        samples = np.zeros((n_samples, dim))
        for i in range(n_samples):
            for j in range(dim):
                samples[i, j] = -5 + (2 * (i + rng.random())) / n_samples
        
        # Evaluate samples
        for x in samples:
            if calls >= budget:
                break
            counting_func(x)
            
        # If no improvement was found (unlikely), fallback to center
        if best_f == np.inf:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        # Phase 2: Exploitation with Nelder-Mead
        # We will run Nelder-Mead starting from best_x.
        # We divide the remaining budget into "restarts" with decreasing spread.
        
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Define restart strategy
        # Spread 1.0, 0.5, 0.1, 0.01
        # Budget allocation: 50%, 30%, 15%, 5%
        spreads = [1.0, 0.5, 0.1, 0.01]
        budget_ratios = [0.5, 0.3, 0.15, 0.05]
        
        current_start = best_x.copy()
        
        for idx, (spread, ratio) in enumerate(zip(spreads, budget_ratios)):
            if calls >= budget or time.time() > deadline:
                break
            
            # Estimate max iterations based on call budget
            # Nelder-Mead uses ~dim+1 calls per iteration
            max_iter = max(10, int((remaining_budget * ratio) / (dim + 1)))
            
            try:
                # Construct initial simplex
                # Center at current_start, spread 'spread'
                initial_simplex = np.zeros((dim + 1, dim))
                initial_simplex[0] = current_start
                for i in range(dim):
                    initial_simplex[i + 1] = current_start
                    initial_simplex[i + 1, i] += spread
                    # Ensure within bounds, though counting_func clips
                    initial_simplex[i + 1] = np.clip(initial_simplex[i + 1], -5, 5)
                
                result = minimize(
                    counting_func,
                    current_start,
                    method='Nelder-Mead',
                    options={
                        'initial_simplex': initial_simplex,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': max_iter,
                        'adaptive': True
                    }
                )
                
                # Update best
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
                
                # Restart from the current best
                current_start = best_x.copy()
                
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
