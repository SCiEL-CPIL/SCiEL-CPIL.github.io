"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time guard: must finish within 300s, use 290s as safety margin
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        # Strategy: Deterministic Sobol sequence for initial global coverage,
        # then (1+1)-ES with adaptive sigma for local refinement.
        
        # Phase 1: Use Sobol sequence for first 80 evaluations to ensure
        # good global coverage and steady early progress.
        from scipy.stats import qmc
        
        n_sobol = min(80, budget)
        sobol_sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol_sampler.random(n_sobol)
        
        for i in range(n_sobol):
            if calls_made >= budget or time.time() >= deadline:
                break
            
            # Map Sobol points from [0,1]^d to [-5,5]^d
            x = -5.0 + 10.0 * sobol_points[i]
            x = np.clip(x, -5.0, 5.0)
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: (1+1)-ES with adaptive sigma and periodic restarts
        if calls_made < budget and best_x is not None:
            # Start ES from the best point found by Sobol sampling
            x = best_x.copy()
            fx = best_f
            sigma = 1.5
            no_improve = 0
            max_no_improve = 25
            restart_count = 0
            max_restarts = 5
            
            while calls_made < budget and time.time() < deadline:
                # Check time more frequently
                if (calls_made % 50 == 0) and time.time() > deadline - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    # Increase sigma slightly on success
                    sigma = min(sigma * 1.2, 8.0)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.75, 0.005)
                
                # Restart if stuck
                if no_improve >= max_no_improve and restart_count < max_restarts:
                    # Structured restart: use additional Sobol-like points
                    # or perturbed best with moderate scale
                    if restart_count % 2 == 0:
                        # Perturb best with moderate scale
                        x = np.clip(best_x + 3.0 * rng.standard_normal(dim), -5, 5)
                    else:
                        # Wider random restart
                        x = rng.uniform(-5, 5, dim)
                    
                    fx = func(x)
                    calls_made += 1
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    sigma = 2.0
                    no_improve = 0
                    restart_count += 1
        
        # Phase 3: Final high-precision polish pass
        if calls_made < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 15
            
            while calls_made < budget and time.time() < deadline:
                if time.time() > deadline - 2:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.7, 0.001)
                
                if no_improve >= max_no_improve:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
