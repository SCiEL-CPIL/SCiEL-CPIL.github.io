import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 55% of budget for global exploration to leave more room for local refinement
        global_budget = int(budget * 0.55)
        
        # Initial stratified sampling for better coverage
        initial_samples = min(8, global_budget // 4)
        for i in range(initial_samples):
            if calls >= budget or time.time() > deadline:
                break
            # Stratified sampling: divide each dimension into 2-3 regions
            x = np.array([rng.uniform(-5 + 10*j/3, -5 + 10*(j+1)/3) for j in range(dim)])
            try:
                fx = safe_func(x)
            except (TimeoutError, StopIteration):
                break
            # Move to best found so far for ES initialization
            if best_x is not None and fx > best_f:
                x = best_x.copy()
                fx = best_f
            else:
                x = x.copy()
                fx = fx
        
        # If no samples were made, fall back to random
        if calls == 0:
            x = rng.uniform(-5, 5, dim)
            try:
                fx = safe_func(x)
            except (TimeoutError, StopIteration):
                return best_x, best_f
        
        # Ensure we have a valid starting point
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
        
        sigma = 1.5
        stagnation = 0
        max_stagnation = 20
        min_sigma = 0.003
        max_sigma = 8.0
        restart_count = 0
        max_restarts = 12
        best_f_history = []
        no_improve_count = 0

        for _ in range(global_budget):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate candidate with occasional large jumps for exploration
            if rng.random() < 0.05:
                # Large exploratory step
                y = x + sigma * 3.0 * rng.standard_normal(dim)
            else:
                y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                # Successful step: increase sigma
                sigma = min(sigma * 1.10, max_sigma)
                stagnation = 0
                no_improve_count = 0
            else:
                # Failed step: decrease sigma
                sigma = max(sigma * 0.90, min_sigma)
                stagnation += 1
                no_improve_count += 1
                
                # Track best for restart decisions
                best_f_history.append(best_f)
                if len(best_f_history) > 50:
                    best_f_history.pop(0)
                
                # If stagnating, restart with better strategy
                if stagnation > max_stagnation and restart_count < max_restarts:
                    restart_count += 1
                    # Adaptive restart: use more diverse strategies as restarts increase
                    diversity_factor = 1.0 + 0.5 * restart_count
                    if best_x is not None:
                        # Mix of strategies: best, perturbed best, random
                        r = rng.random()
                        if r < 0.25:
                            x = best_x.copy()
                            fx = best_f
                        elif r < 0.6:
                            # Perturb best with increasing diversity
                            x = best_x + diversity_factor * 2.0 * rng.standard_normal(dim)
                        else:
                            x = rng.uniform(-5, 5, dim)
                        x = np.clip(x, -5, 5)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    # Reset parameters for fresh start
                    sigma = 3.0
                    stagnation = 0
                    max_stagnation = max(10, max_stagnation // 2)
                    no_improve_count = 0
                    best_f_history = []

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 15:
                # Nelder-Mead: use a fraction of remaining budget
                max_iter = min(remaining_calls // (dim + 3), 50)
                max_iter = max(8, max_iter)
                
                try:
                    # Initialize simplex around best_x with adaptive spread
                    x0 = best_x.copy()
                    # Create initial simplex with small perturbations
                    initial_simplex = [x0]
                    for i in range(dim):
                        perturbation = x0.copy()
                        perturbation[i] += 0.1
                        initial_simplex.append(np.clip(perturbation, -5, 5))
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True,
                            'initial_simplex': initial_simplex
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final High-Precision Polish with adaptive (1+1)-ES
        # Uses success-rate-based step size adaptation for better convergence
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                x = best_x.copy()
                fx = best_f
                
                # Adaptive sigma: start small, increase if making progress
                sigma = 0.02
                success_window = []
                window_size = 8
                target_success_rate = 0.25
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    
                    success = fy < fx
                    success_window.append(1 if success else 0)
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    
                    if success:
                        x, fx = y, fy
                    
                    # Adapt sigma based on success rate over recent window
                    if len(success_window) >= window_size:
                        recent_rate = sum(success_window) / len(success_window)
                        if recent_rate > target_success_rate:
                            sigma = min(sigma * 1.15, 0.5)
                        elif recent_rate < target_success_rate * 0.3:
                            sigma = max(sigma * 0.85, 1e-7)
                        else:
                            sigma = max(sigma * 0.98, 1e-7)
                    
                    # If sigma becomes very small, we've likely converged
                    if sigma < 1e-6:
                        break

        return best_x, best_f
