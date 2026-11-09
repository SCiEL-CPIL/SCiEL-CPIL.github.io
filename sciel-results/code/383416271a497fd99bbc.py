"""Model-and-Optimize Optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using multi-start Adaptive Simulated Annealing (ASA).
    ASA uses temperature-controlled random perturbations to escape local minima,
    making it effective for rotated landscapes where gradient-free methods struggle.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 20s safety margin
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        def asa_optimize(start_x, max_calls):
            """
            Run Adaptive Simulated Annealing from start_x.
            Returns the best x found.
            """
            nonlocal calls_made, global_best_x, global_best_f
            
            if calls_made >= budget or time.time() > deadline:
                return start_x
            
            x = start_x.copy()
            f_x = call_func(x)
            best_x = x.copy()
            best_f = f_x
            
            # Temperature parameters
            T_initial = 5.0  # Initial temperature (scale of perturbations)
            T_min = 0.01     # Minimum temperature
            alpha = 0.98     # Cooling rate
            T = T_initial
            
            # Track progress for adaptive temperature
            no_improve_count = 0
            patience = 20
            
            for step in range(max_calls):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate ASA perturbation
                # Direction: random unit vector
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                # Step size based on temperature
                step_size = T
                
                # Propose new point
                x_new = x + step_size * direction
                
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = call_func(x_new)
                
                # Acceptance criterion (Metropolis)
                if f_new < f_x:
                    # Always accept improvements
                    x = x_new
                    f_x = f_new
                    no_improve_count = 0
                    
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                else:
                    # Accept with probability exp(-(f_new - f_x) / T)
                    delta_f = f_new - f_x
                    if delta_f < 1e-10:
                        accept_prob = 1.0
                    else:
                        accept_prob = np.exp(-delta_f / max(T, 1e-10))
                    
                    if np.random.random() < accept_prob:
                        x = x_new
                        f_x = f_new
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                
                # Adaptive cooling
                # Cool faster if no improvement for a while
                if no_improve_count >= patience:
                    T *= 0.9  # Extra cooling
                    no_improve_count = 0
                
                T *= alpha
                T = max(T, T_min)
            
            return best_x

        # --- Phase 1: Multi-start ASA ---
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Start points: center, random, and some structured
        starts = [np.zeros(dim)]
        
        # Add random starts
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Add some structured starts (grid-like)
        for _ in range(5):
            start = rng.uniform(-3, 3, dim)
            starts.append(start)
        
        # Shuffle starts for diversity
        rng.shuffle(starts)
        
        # Allocate budget: use ~80% for ASA, leave 20% for refinement
        asa_budget = int(budget * 0.8)
        calls_per_start = max(10, asa_budget // len(starts))
        
        candidates = []
        
        for start_x in starts:
            if time.time() > deadline or calls_made >= budget:
                break
            
            remaining = budget - calls_made
            if remaining < 5:
                break
            
            # Limit calls for this start
            max_calls_for_start = min(calls_per_start, remaining)
            
            try:
                best_x = asa_optimize(start_x, max_calls_for_start)
                if best_x is not None:
                    candidates.append(best_x)
            except Exception:
                pass
        
        # --- Phase 2: Final refinement with Nelder-Mead on best candidate ---
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            from scipy.optimize import minimize
            
            # Refine the global best
            remaining_calls = budget - calls_made
            if remaining_calls > 10:
                try:
                    res = minimize(
                        call_func, 
                        global_best_x.copy(), 
                        method='Nelder-Mead', 
                        options={
                            'xatol': 1e-8, 
                            'fatol': 1e-10, 
                            'maxfev': min(50, remaining_calls), 
                            'maxiter': 100
                        }
                    )
                    if res.fun < global_best_f:
                        global_best_f = res.fun
                        global_best_x = res.x.copy()
                except Exception:
                    pass
            
            # Also try refining a few other good candidates
            if calls_made < budget and time.time() < deadline:
                # Find unique candidates
                unique_candidates = []
                for c in candidates:
                    if not any(np.linalg.norm(c - uc) < 0.5 for uc in unique_candidates):
                        unique_candidates.append(c)
                
                # Sort by distance to global best (prefer closer ones)
                unique_candidates.sort(key=lambda c: np.linalg.norm(c - global_best_x))
                
                for c in unique_candidates[:2]:
                    if time.time() > deadline or calls_made >= budget:
                        break
                    
                    remaining_calls = budget - calls_made
                    if remaining_calls < 10:
                        break
                    
                    try:
                        res = minimize(
                            call_func, 
                            c.copy(), 
                            method='Nelder-Mead', 
                            options={
                                'xatol': 1e-8, 
                                'fatol': 1e-10, 
                                'maxfev': min(30, remaining_calls), 
                                'maxiter': 50
                            }
                        )
                        if res.fun < global_best_f:
                            global_best_f = res.fun
                            global_best_x = res.x.copy()
                    except Exception:
                        pass
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
