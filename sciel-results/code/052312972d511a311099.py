"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Adaptive (1+1)-ES with restarts
        # For Lunacek's function, we have multiple basins. We use restarts
        # to escape local minima.

        # Strategy: run (1+1)-ES from random starts, restart if no improvement
        # for a while, and keep the best solution found.

        # We divide the budget among multiple restarts.
        # Each restart gets a portion of the budget.
        
        # Let's use a scheme where we do several restarts.
        # Each restart: run (1+1)-ES for some iterations, then restart from
        # a new random point if stuck, or from a perturbation of the best.
        
        # Key insight: Lunacek has a global minimum and a local minimum.
        # The local minimum is at x* = (-1.5, ..., -1.5) approximately (in the
        # transformed space). We need to explore widely enough to find the global one.
        
        # Let's use a strategy:
        # 1. Start with a few random restarts to get a good initial solution
        # 2. Then do a focused (1+1)-ES with occasional large jumps
        
        # Actually, let's keep it simpler: (1+1)-ES with restarts when stuck.
        
        num_restarts_max = 10  # maximum number of restarts
        restart_threshold = 50  # no improvement for this many iterations -> restart
        
        for restart in range(num_restarts_max):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            
            # For the first restart, start random. For subsequent, start near best
            # with some perturbation, or randomly.
            if restart == 0:
                x = rng.uniform(-5, 5, n)
            elif best_x is not None:
                # Perturb the best solution
                sigma_init = 2.0
                x = np.clip(best_x + sigma_init * rng.standard_normal(n), -5, 5)
                # With some probability, do a fully random restart
                if rng.random() < 0.3:
                    x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-5, 5, n)
            
            fx = safe_func(x)
            sigma = 2.0
            no_improve = 0
            
            while calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.3
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                if no_improve >= restart_threshold:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
