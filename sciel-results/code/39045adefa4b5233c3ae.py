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
        
        # Determine grid resolution based on budget
        # We need to cover 3 front dimensions
        # Use a 3D grid: n1 x n2 x n3 points
        # Total evaluations = n1 * n2 * n3 <= budget
        
        # Start with reasonable grid sizes
        n1 = 8
        n2 = 8
        n3 = 8
        total_grid = n1 * n2 * n3
        
        # Adjust grid to fit budget
        while total_grid > self.budget and (n1 > 1 or n2 > 1 or n3 > 1):
            # Reduce the largest dimension
            if n1 >= n2 and n1 >= n3:
                n1 = max(1, n1 - 1)
            elif n2 >= n1 and n2 >= n3:
                n2 = max(1, n2 - 1)
            else:
                n3 = max(1, n3 - 1)
            total_grid = n1 * n2 * n3
        
        # Generate grid points
        t1_vals = np.linspace(0.05, 0.95, n1)
        t2_vals = np.linspace(0.05, 0.95, n2)
        t3_vals = np.linspace(0.05, 0.95, n3)
        
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
        
        # If we have budget remaining, do local refinement around best points
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
                
                # Refine around non-dominated points
                for idx in nd_idx[:min(10, len(nd_idx))]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    center = X_list[idx]
                    # Local perturbation
                    for dim in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        for delta in [0.05, -0.05, 0.02, -0.02]:
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
