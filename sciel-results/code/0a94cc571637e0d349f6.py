"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for BBOB f15 (Rotated Rastrigin).
    
    Strategy:
    1. Sobol Quasi-Random Exploration: Systematically cover the domain using a low-discrepancy sequence.
    2. Adaptive (1+1)-ES Refinement: Intensively refine the best point found using an adaptive step-size (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Sobol Quasi-Random Exploration ---
        # Use a Sobol sequence to generate diverse, uniformly distributed points.
        # This is more efficient than random sampling for covering the space.
        
        # Determine how many points to use for exploration.
        # We want to leave enough budget for refinement.
        # Let's allocate 30% of the budget to exploration, but at least 50 points.
        n_exploration = max(50, int(budget * 0.3))
        
        # Generate Sobol points
        sobol = qmc.Sobol(d=dim, scramble=True)
        # We need 2^k points for Sobol. Find the next power of 2 >= n_exploration
        k = int(np.ceil(np.log2(n_exploration)))
        n_sobol = 2 ** k
        
        # Generate in [0, 1]^dim
        points_unit = sobol.random(n_sobol)
        
        # Scale to [-5, 5]^dim
        points = points_unit * 10.0 - 5.0
        
        # Evaluate Sobol points
        for point in points:
            if calls >= budget or time.time() > deadline:
                break
            eval_func(point)
            
        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Refine the best point found in Phase 1.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma for local search
            sigma = 0.5
            window = 10
            recent = [False] * window
            idx = 0
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.15
                    elif sr < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
