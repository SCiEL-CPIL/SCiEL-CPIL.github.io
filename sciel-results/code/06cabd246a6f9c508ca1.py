"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with deadline guard."""

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
        
        # Phase 1: Diverse initial sampling - mix of random, grid, and center
        n_initial = min(25, self.budget)
        # First try origin and a few structured points
        structured_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0][:self.dim]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0][:self.dim]),
        ]
        for pt in structured_points[:min(5, n_initial)]:
            if time.time() > deadline:
                break
            x = pt.copy()
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Then random samples
        for i in range(n_initial - min(5, n_initial)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = call_func(best_x)
        
        # Phase 2: Multiple restarts with (1+1)-ES and occasional large jumps
        remaining_budget = self.budget - calls_made
        n_restarts = min(5, max(1, remaining_budget // 70))
        budget_per_restart = remaining_budget // n_restarts
        
        for restart in range(n_restarts):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Start from current best with perturbation
            if restart == 0:
                x = best_x.copy()
            else:
                # Larger perturbation for later restarts to escape local minima
                perturbation_scale = 0.5 + 0.3 * restart
                x = best_x + perturbation_scale * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
            
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            sigma = 1.0
            steps_since_improvement = 0
            for step in range(budget_per_restart):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Occasional large jump to escape local minima
                if step > 0 and step % 50 == 0 and steps_since_improvement > 20:
                    x = best_x + 2.0 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    steps_since_improvement = 0
                    continue
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = call_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.25, 5.0)
                    steps_since_improvement = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    sigma = max(sigma, 0.005)
                    steps_since_improvement += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
