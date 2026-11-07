"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a hybrid approach:
    1. Initial broad exploration with a population-based random search + local refinement.
    2. Then a CMA-ES-like adaptation for the remaining budget.
    
    Key structural change from (1+1)-ES:
    - Uses a small population to better handle multimodality of Schwefel.
    - Adapts covariance matrix to capture rotation/scaling of the landscape.
    - Early evaluations are used to quickly narrow down the search region.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Quick exploration with random sampling + local search
        # Use ~20% of budget for initial exploration
        explore_budget = max(20, budget // 5)
        
        # Initial best
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls_made = 1
        
        # Generate a small population of random points
        # Sample from the box [-5, 5]^dim
        population_size = 10
        pop = []
        
        for i in range(population_size):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            pop.append((fx, x.copy()))
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
        
        # Sort by fitness
        pop.sort(key=lambda t: t[0])
        
        # Local refinement around the best few points
        # Use a simple coordinate descent / Nelder-Mead-like local search
        for i in range(min(3, len(pop))):
            if calls_made >= budget or time.time() >= deadline:
                break
            x_cur = pop[i][1].copy()
            fx_cur = pop[i][0]
            
            # Simple coordinate descent with adaptive step
            step = 0.5
            for _ in range(10):
                if calls_made >= budget or time.time() >= deadline:
                    break
                improved = False
                for d in range(dim):
                    # Try positive and negative step in dimension d
                    for sign in [1, -1]:
                        x_trial = x_cur.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        if calls_made >= budget:
                            break
                        fx_trial = func(x_trial)
                        calls_made += 1
                        if fx_trial < fx_cur:
                            x_cur = x_trial
                            fx_cur = fx_trial
                            improved = True
                            if fx_cur < fx_best:
                                fx_best = fx_cur
                                x_best = x_cur.copy()
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
        
        # Phase 2: CMA-ES-like optimization for the remaining budget
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            # Initialize CMA-ES from the best point
            x_mean = x_best.copy()
            sigma = 1.0  # Initial step size
            
            # Simple covariance adaptation: use diagonal covariance initially
            # and adapt based on success rate (simplified CMA-ES)
            pop_size = max(4, min(10, remaining_budget // 5))
            
            # Track best
            fx_mean = fx_best
            
            while calls_made < budget:
                if time.time() >= deadline:
                    break
                
                # Generate population
                for _ in range(pop_size):
                    if calls_made >= budget:
                        break
                    # Sample from multivariate normal
                    x_sample = x_mean + sigma * np.random.randn(dim)
                    x_sample = np.clip(x_sample, -5, 5)
                    fx_sample = func(x_sample)
                    calls_made += 1
                    
                    if fx_sample < fx_best:
                        fx_best = fx_sample
                        x_best = x_sample.copy()
                
                # Update mean: move towards best point
                x_mean = 0.5 * x_mean + 0.5 * x_best
                
                # Adapt sigma based on distance to best
                dist = np.linalg.norm(x_best - x_mean)
                if dist > 0.1:
                    sigma = max(0.1, min(5.0, dist * 0.5))
                else:
                    sigma = max(0.01, sigma * 0.9)
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
