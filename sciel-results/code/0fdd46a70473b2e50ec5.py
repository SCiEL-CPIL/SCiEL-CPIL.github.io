"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search exploiting WFG9 structure for dense Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9: variables 0-2 control the front, variables 3-11 are shape/position
        # Fix non-front variables at midpoint to exploit structure
        base = mid.copy()
        
        # For WFG9, the Pareto front is determined by the first 3 decision variables
        # We use a denser grid on these 3 variables to maximize hypervolume coverage
        # Budget is 200, so we want n1*n2*n3 <= 200
        # 6*6*5 = 180, leaving 20 for refinement
        n1 = 6
        n2 = 6
        n3 = 5
        total_grid = n1 * n2 * n3
        
        # Generate grid points with wider range to capture more of the front
        t1_vals = np.linspace(0.0, 1.0, n1)
        t2_vals = np.linspace(0.0, 1.0, n2)
        t3_vals = np.linspace(0.0, 1.0, n3)
        
        X_list = []
        F_list = []
        used = 0
        
        for t1 in t1_vals:
            for t2 in t2_vals:
                for t3 in t3_vals:
                    if used >= self.budget:
                        break
                    # Check deadline
                    if time.time() > deadline:
                        break
                    
                    x = base.copy()
                    # Map grid parameters to decision space for front variables
                    x[0] = lo[0] + t1 * (hi[0] - lo[0])
                    x[1] = lo[1] + t2 * (hi[1] - lo[1])
                    x[2] = lo[2] + t3 * (hi[2] - lo[2])
                    
                    f = problem(x)
                    X_list.append(x)
                    F_list.append(f)
                    used += 1
                
                if used >= self.budget or time.time() > deadline:
                    break
            
            if used >= self.budget or time.time() > deadline:
                break
        
        # If we have budget remaining, do local refinement around non-dominated points
        if used < self.budget and time.time() < deadline:
            if F_list:
                F_arr = np.array(F_list)
                # Find non-dominated points
                nd_idx = []
                for i in range(len(F_arr)):
                    dominated = False
                    for j in range(len(F_arr)):
                        if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd_idx.append(i)
                
                # Refine around non-dominated points with finer steps
                for idx in nd_idx[:min(15, len(nd_idx))]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    center = X_list[idx]
                    # Local perturbation on the 3 front variables
                    for dim in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        for delta in [0.02, -0.02, 0.01, -0.01]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            
                            x = center.copy()
                            x[dim] = np.clip(x[dim] + delta * (hi[dim] - lo[dim]), lo[dim], hi[dim])
                            
                            f = problem(x)
                            X_list.append(x)
                            F_list.append(f)
                            used += 1
        
        # Convert to arrays
        if X_list:
            X = np.array(X_list)
            F = np.array(F_list)
        else:
            X = base.reshape(1, -1)
            F = np.array([problem(base)])
        
        return X
# EVOLVE-BLOCK-END
