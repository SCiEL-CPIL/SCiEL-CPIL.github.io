"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid Optimizer: Multi-Start Simulated Annealing (Global) + Nelder-Mead (Local).
    
    Strategy:
    1. Split budget into K global SA runs.
    2. Each SA run explores the landscape, accepting worse moves probabilistically to escape local minima.
    3. Collect the best K points.
    4. Use remaining budget to run Nelder-Mead local refinement on these best points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        # Allocation: 70% for Global SA, 30% for Local NM
        global_budget = int(total_budget * 0.7)
        local_budget = total_budget - global_budget
        
        # Number of SA restarts
        num_restarts = 10
        calls_per_sa = max(10, global_budget // num_restarts)
        
        best_overall_x = None
        best_overall_f = np.inf
        best_candidates = []
        
        calls_used = 0
        
        # --- Global Phase: Multi-Start Simulated Annealing ---
        for r in range(num_restarts):
            # Random start
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            
            current_x = x.copy()
            current_f = f
            best_x_in_run = x.copy()
            best_f_in_run = f
            
            # SA Parameters
            T = 10.0
            cooling_rate = 0.95
            step_size = 1.5
            
            # Run SA for allocated calls
            for step in range(calls_per_sa):
                if calls_used >= global_budget:
                    break
                
                # Generate neighbor
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                # Adaptive step size: shrink slightly each step
                new_x = current_x + step_size * direction
                # Clip to box
                new_x = np.clip(new_x, -5, 5)
                
                new_f = func(new_x)
                calls_used += 1
                
                # Acceptance Criterion
                delta_f = new_f - current_f
                if delta_f < 0:
                    # Always accept improvement
                    current_x = new_x
                    current_f = new_f
                    if current_f < best_f_in_run:
                        best_f_in_run = current_f
                        best_x_in_run = current_x.copy()
                else:
                    # Probabilistic acceptance of worse solution
                    if T > 1e-8:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            current_x = new_x
                            current_f = new_f
                
                # Cool down
                T *= cooling_rate
                step_size *= 0.99 # Slowly shrink step size
            
            if best_f_in_run < best_overall_f:
                best_overall_f = best_f_in_run
                best_overall_x = best_x_in_run.copy()
            
            best_candidates.append((best_f_in_run, best_x_in_run))
            
            if calls_used >= global_budget:
                break
        
        # --- Local Phase: Nelder-Mead Refinement ---
        # Sort candidates by fitness
        best_candidates.sort(key=lambda x: x[0])
        
        # Use remaining budget for local optimization
        # Nelder-Mead is efficient. We can run it on the top few candidates.
        # Each NM run might take ~50-100 calls depending on convergence.
        # We have local_budget left.
        
        # Limit number of local runs to avoid exceeding budget
        # Assume ~50 calls per NM run on average.
        max_local_runs = max(1, local_budget // 50)
        local_runs = min(max_local_runs, len(best_candidates))
        
        for i in range(local_runs):
            if calls_used >= total_budget:
                break
            
            x0, f0 = best_candidates[i]
            
            # Define a wrapper to track calls and stop if budget exceeded
            calls_in_nm = 0
            max_nm_calls = total_budget - calls_used
            
            def nm_func(x):
                nonlocal calls_in_nm
                if calls_in_nm >= max_nm_calls:
                    return f0 # Return current best to stop effectively
                fx = func(x)
                calls_in_nm += 1
                calls_used += 1
                return fx
            
            try:
                # Nelder-Mead is derivative-free
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_nm_calls, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < best_overall_f:
                    best_overall_f = res.fun
                    best_overall_x = res.x.copy()
                    
            except Exception:
                pass
        
        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
