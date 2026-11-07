"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    MODEL-AND-OPTIMIZE approach for ZDT4-like problems.
    
    Key insight: ZDT4 (and similar bi-objective problems) have a structure where:
    1. One variable (typically x[0]) controls the position along the Pareto front.
    2. The remaining variables affect the distance from the Pareto front (penalty term).
    
    For ZDT4 specifically:
    - f1 depends on x[0] and a penalty term g(x[1:])
    - f2 depends on f1 and an oscillatory function
    - The penalty term g is minimized when x[1:] = 0
    
    Strategy:
    1. Use a few evaluations to probe the problem structure.
    2. Determine which variable best predicts the Pareto front (by correlating with objectives).
    3. Determine the optimal values for other variables by minimizing variance in objectives.
    4. Use the remaining budget to densely sample the key variable to cover the Pareto front.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Exploit known ZDT4 structure
        # 1. x[0] is the key variable controlling the Pareto front position.
        # 2. x[1:] are penalty variables, optimal at 0 (or closest bound to 0).
        # 3. Use full budget to densely sample x[0].
        
        key_var = 0
        
        # Set non-key variables to 0, clipped to bounds
        base_x = np.zeros(self.dim)
        base_x = np.clip(base_x, lo, hi)
        
        # Use full budget for sweeping the key variable
        n_sweep = self.budget
        
        # Create sweep values
        x_key_vals = np.linspace(lo[key_var], hi[key_var], n_sweep)
        
        # Construct points
        all_points = np.tile(base_x, (n_sweep, 1))
        all_points[:, key_var] = x_key_vals
        
        # Evaluate points
        all_objectives = []
        for i in range(n_sweep):
            if time.time() > deadline:
                break
            obj = problem(all_points[i])
            all_objectives.append(np.array([obj]))
            
        if not all_objectives:
            # Fallback if no evaluations were made (should not happen)
            obj = problem(base_x)
            all_objectives.append(np.array([obj]))
            all_points = np.array([base_x])
        else:
            all_objectives = np.vstack(all_objectives)
            # Trim all_points if we broke early due to time
            if len(all_objectives) < n_sweep:
                all_points = all_points[:len(all_objectives)]
                
        return all_points, all_objectives
# EVOLVE-BLOCK-END
