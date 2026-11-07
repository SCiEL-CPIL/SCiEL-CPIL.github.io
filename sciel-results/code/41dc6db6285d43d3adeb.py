"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Optimizer: Population Init + (1+1)-ES with Restarts + Rastrigin Snapping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        def time_ok():
            return time.time() < deadline

        calls = 0
        
        # Phase 1: Population-based exploration
        # Use a modest population to cover the search space broadly
        pop_size = min(20, budget // 10)
        if pop_size < 10:
            pop_size = 10
            
        best_x = None
        best_fx = np.inf
        
        # Evaluate initial population
        for i in range(pop_size):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no calls were made
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        x = best_x.copy()
        fx = best_fx
        sigma = 1.5
        no_improve_count = 0
        max_no_improve = 30 # Restart threshold
        
        while calls < budget and time_ok():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1  # Increase step size on success
                no_improve_count = 0
            else:
                sigma *= 0.9  # Decrease step size on failure
                no_improve_count += 1
                
            # Stagnation handling: If stuck, try a larger jump or restart sigma
            if no_improve_count >= max_no_improve:
                if sigma < 1e-3:
                    # If step size is tiny and stuck, likely in a local min.
                    # Try a random restart from the best point found so far with a large kick
                    x = best_x + rng.uniform(-2, 2, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    else:
                        x = best_x.copy()
                        fx = best_fx
                    sigma = 1.0
                    no_improve_count = 0
                else:
                    # Just reset sigma to a moderate value
                    sigma = max(sigma * 2.0, 0.5)
                    no_improve_count = 0
            
            if sigma < 1e-8:
                break

        # Phase 3: Rastrigin-aware refinement
        # Rastrigin local minima are at integer coordinates.
        # Try snapping each coordinate to nearest integer, floor, and ceil.
        if time_ok() and calls < budget - 10:
            for coord in range(dim):
                if not time_ok() or calls >= budget - 5:
                    break
                
                candidates = []
                # Nearest integer
                r = np.round(x[coord])
                candidates.append(r)
                # Floor
                candidates.append(np.floor(x[coord]))
                # Ceil
                candidates.append(np.ceil(x[coord]))
                
                for cand in candidates:
                    if not time_ok() or calls >= budget:
                        break
                    x_test = x.copy()
                    x_test[coord] = cand
                    x_test = np.clip(x_test, -5, 5)
                    fx_test = func(x_test)
                    calls += 1
                    if fx_test < fx:
                        x = x_test
                        fx = fx_test

        # Phase 4: Fine-grained local search
        # Use a very small sigma for final polishing
        if time_ok() and calls < budget:
            sigma_small = 0.05
            while calls < budget and time_ok():
                y = np.clip(x + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_small *= 1.05
                else:
                    sigma_small *= 0.8
                    if sigma_small < 1e-9:
                        break

        return x, fx

# EVOLVE-BLOCK-END
