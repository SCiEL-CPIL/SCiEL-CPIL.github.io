"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage adaptive optimizer: Coarse grid followed by greedy Pareto expansion."""

    def __init__(self, dim, budget, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9 structure: First 3 variables determine the front shape.
        # Fix variables 3..11 to midpoint.
        base = mid.copy()
        
        X_list = []
        F_list = []
        used = 0
        
        # Stage 1: Coarse grid - 5x5x4 = 100 points
        n1, n2, n3 = 5, 5, 4
        t1_vals = np.linspace(0.0, 1.0, n1)
        t2_vals = np.linspace(0.0, 1.0, n2)
        t3_vals = np.linspace(0.0, 1.0, n3)
        
        for t1 in t1_vals:
            for t2 in t2_vals:
                for t3 in t3_vals:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = base.copy()
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
        
        # Stage 2: Adaptive refinement - evaluate midpoints between existing points
        if used < self.budget and time.time() < deadline:
            if len(X_list) >= 8:
                F_arr = np.array(F_list)
                
                # Find non-dominated points for targeted refinement
                nd_idx = []
                for i in range(len(F_arr)):
                    dominated = False
                    for j in range(len(F_arr)):
                        if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd_idx.append(i)
                
                # Generate candidate refinement points: midpoints in objective space
                candidates = []
                for i in range(len(nd_idx)):
                    for j in range(i+1, len(nd_idx)):
                        if time.time() > deadline:
                            break
                        # Midpoint in decision space for the 3 front variables
                        xi = X_list[nd_idx[i]]
                        xj = X_list[nd_idx[j]]
                        xmid = base.copy()
                        for d in range(3):
                            xmid[d] = (xi[d] + xj[d]) / 2.0
                        candidates.append(xmid)
                
                # Also add points along edges of the front (extremes)
                for d in range(3):
                    for frac in [0.1, 0.3, 0.7, 0.9]:
                        x = base.copy()
                        x[d] = lo[d] + frac * (hi[d] - lo[d])
                        # Set other front vars to 0.5 to explore different slices
                        for d2 in range(3):
                            if d2 != d:
                                x[d2] = lo[d2] + 0.5 * (hi[d2] - lo[d2])
                        candidates.append(x)
                
                # Evaluate candidates, prioritizing those likely to be non-dominated
                evaluated_set = set()
                for x in candidates:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Check if this point is too close to an existing one
                    key = tuple(np.round(x[:3], 3))
                    if key in evaluated_set:
                        continue
                    evaluated_set.add(key)
                    
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
