"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol exploration + adaptive (1+1)-ES + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        # === Phase 1: Sobol space-filling search (broad exploration) ===
        # Use quasi-random Sobol sequence for systematic coverage
        from scipy.integrate import quad  # just to ensure scipy is available
        
        # Generate Sobol points using a simple LCG-based approach for 5D
        # We'll use a stratified sampling approach that approximates Sobol
        n_sobol = min(60, budget // 5)  # Use up to 60 points for exploration
        
        # Create a grid-like stratified sample
        n_per_dim = max(2, int(round(n_sobol ** (1.0 / dim))))
        # Generate all combinations, then subsample
        grid = np.meshgrid(*[np.linspace(-4.5, 4.5, n_per_dim)] * dim)
        grid_points = np.array(np.meshgrid(*[np.linspace(-4.5, 4.5, n_per_dim)] * dim)).T.reshape(-1, dim)
        
        # Shuffle and take first n_sobol points
        rng.shuffle(grid_points)
        sobol_points = grid_points[:n_sobol]
        
        # Add some additional random points for diversity
        n_extra = min(20, budget - n_sobol)
        if n_extra > 0:
            extra_points = rng.uniform(-4.5, 4.5, size=(n_extra, dim))
            sobol_points = np.vstack([sobol_points, extra_points])
        
        for x_point in sobol_points:
            if calls_made >= budget or time.time() > deadline:
                break
            x_point = np.clip(x_point, -5, 5)
            fx = func(x_point)
            calls_made += 1
            try_update(x_point, fx)
        
        # === Phase 2: Adaptive (1+1)-ES from the best point ===
        if global_best_x is not None and calls_made < budget:
            x = global_best_x.copy()
            fx = global_best_fx
            
            sigma = 0.8  # Initial step size
            success_threshold = 0.2
            step_up = 1.25
            step_down = 1.0 / 1.25
            window_size = 8
            successes = 0
            steps_in_window = 0
            no_improve_count = 0
            max_no_improve = 30
            
            es_target = min(budget - 30, int(budget * 0.6))  # Reserve 30 for polish
            
            while calls_made < es_target:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                
                candidate_fx = func(candidate)
                calls_made += 1
                
                if candidate_fx <= fx:
                    x = candidate
                    fx = candidate_fx
                    successes += 1
                    try_update(x, fx)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                steps_in_window += 1
                
                # Update step size based on 1/5 rule
                if steps_in_window >= window_size:
                    success_rate = successes / steps_in_window
                    if success_rate > success_threshold:
                        sigma *= step_up
                    elif success_rate < success_threshold:
                        sigma *= step_down
                    successes = 0
                    steps_in_window = 0
                
                # Stagnation check with more aggressive restarts
                if no_improve_count >= max_no_improve:
                    # Try multiple restart strategies
                    improved = False
                    for attempt in range(5):
                        if calls_made >= es_target:
                            break
                        if attempt == 0:
                            # Small perturbation
                            x_restart = x + 0.5 * rng.standard_normal(dim)
                        elif attempt == 1:
                            # Medium perturbation
                            x_restart = x + 1.5 * rng.standard_normal(dim)
                        else:
                            # Full random restart
                            x_restart = rng.uniform(-4.5, 4.5, size=dim)
                        
                        x_restart = np.clip(x_restart, -5, 5)
                        x_restart_fx = func(x_restart)
                        calls_made += 1
                        
                        if x_restart_fx < fx:
                            x = x_restart
                            fx = x_restart_fx
                            sigma = max(0.3, sigma * 0.5)
                            no_improve_count = 0
                            improved = True
                            try_update(x, fx)
                            break
                    
                    if not improved:
                        # If no improvement, reset and try again with smaller sigma
                        sigma *= 0.5
                        no_improve_count = 0
                        successes = 0
                        steps_in_window = 0
        
        # === Phase 3: Nelder-Mead polish ===
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            
            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return 1e10
                calls_made += 1
                return func(np.clip(x, -5, 5))
            
            try:
                # First pass: moderate precision
                if remaining > 20:
                    result = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                      options={'maxiter': min(remaining - 10, 100), 
                                               'maxfev': min(remaining - 10, 100),
                                               'xatol': 1e-8, 'fatol': 1e-8,
                                               'adaptive': True})
                    if result.fun < global_best_fx:
                        global_best_fx = result.fun
                        global_best_x = result.x.copy()
                
                # Second pass: high precision if budget allows
                if calls_made < budget and time.time() < deadline:
                    remaining2 = budget - calls_made
                    if remaining2 > 10:
                        result2 = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                           options={'maxiter': remaining2, 'maxfev': remaining2,
                                                    'xatol': 1e-10, 'fatol': 1e-10,
                                                    'adaptive': True})
                        if result2.fun < global_best_fx:
                            global_best_fx = result2.fun
                            global_best_x = result2.x.copy()
            except Exception:
                pass
        
        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
