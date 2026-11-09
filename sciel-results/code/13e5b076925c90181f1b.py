"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with multi-restart on stagnation and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin
        
        rng = np.random.default_rng(0)
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx
        
        # Phase 1: Diverse initial sampling
        n_initial = min(20, self.budget)
        # Try origin first
        x0 = np.zeros(self.dim)
        fx0 = call_func(x0)
        best_fx = fx0
        best_x = x0.copy()
        
        # Random sampling around the space
        for i in range(1, n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: (1+lambda)-ES with adaptive sigma
        lambda_ = 4
        sigma = 1.0
        x = best_x.copy()
        fx = best_fx
        no_improve_count = 0
        max_no_improve = 40
        restart_count = 0
        max_restarts = 5
        
        while calls_made < self.budget and time.time() < deadline:
            # Generate lambda offspring
            offspring = []
            for i in range(lambda_):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = call_func(y)
                offspring.append((fy, y))
            
            if not offspring:
                break
            
            # Select best offspring
            offspring.sort(key=lambda t: t[0])
            best_fy, best_y = offspring[0]
            
            if best_fy < fx:
                x, fx = best_y, best_fy
                sigma *= 1.3
                no_improve_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                sigma = max(sigma, 0.001)
                no_improve_count += 1
                
                # Restart if stuck
                if no_improve_count >= max_no_improve:
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    # Perturb from current best
                    x = best_x + 0.5 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma = 0.8
                    no_improve_count = 0
        
        # Phase 3: Final polish - fine-tune around best solution
        if calls_made < self.budget and time.time() < deadline:
            sigma = 0.05
            x = best_x.copy()
            fx = best_fx
            for step in range(min(50, self.budget - calls_made)):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
