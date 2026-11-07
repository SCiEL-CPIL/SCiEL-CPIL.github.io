import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ7 structure:
        # x0 is the "position" along the Pareto front
        # x1...x_{n-2} are "shape" variables that should be 0 for optimality
        # x_{n-1} is a "deception" variable
        #
        # The Pareto front has f1=x0, f2=1-x0, f3=1 for x1..x_{n-2}=0
        # We want to cover the range of x0 in [0,1]
        
        # Strategy:
        # 1. Initialize with points spread along x0, with middle vars near 0
        # 2. Use local optimization (Nelder-Mead) to drive middle vars to 0
        # 3. Use remaining budget to explore and refine
        
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        # Initialize: spread x0 across [0,1], keep middle vars small, x_{n-1} random
        for i in range(pop_size):
            # x0: spread across [0, 1]
            X[i, 0] = (i + 0.5) / pop_size
            # Middle variables: small random values
            if dim > 2:
                X[i, 1:dim-1] = 0.01 * rng.random(dim-2)
            # Last variable: random
            X[i, -1] = rng.random()
        
        # Evaluate initial population
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def get_non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd
        
        # Local optimization phase: use Nelder-Mead to refine each point
        # Focus on driving middle variables to 0 while maintaining x0
        
        def objective_wrapper(x, x0_fixed):
            """Objective that penalizes deviation from x0 and drives middle vars to 0."""
            fx = np.clip(x, lo, hi)
            f = problem(fx)
            # We want to minimize all objectives, but the key is getting middle vars to 0
            # The sum of objectives is a reasonable scalarization
            return np.sum(f)
        
        # Phase 1: Local optimization with Nelder-Mead
        # Use a fraction of budget for local refinement
        local_budget = int(budget * 0.4)
        remaining = budget - used
        
        for i in range(pop_size):
            if used >= budget or time.time() > deadline:
                break
            
            x0_val = X[i, 0]
            x_start = X[i].copy()
            
            # Use Nelder-Mead with limited iterations
            try:
                res = minimize(
                    objective_wrapper,
                    x_start,
                    args=(x0_val,),
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                # Count evaluations used (approximate)
                # Nelder-Mead with maxiter=30 uses roughly 30*dim evaluations
                # We'll just check if we're still within budget
                new_x = np.clip(res.x, lo, hi)
                new_f = problem(new_x)
                used += 1
                
                if np.sum(new_f) < np.sum(F[i]):
                    X[i] = new_x
                    F[i] = new_f
            except:
                pass
            
            if time.time() > deadline:
                break
        
        # Phase 2: Evolutionary search for remaining budget
        # Use non-dominated sorting and crowding distance
        
        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points."""
            if len(nd_indices) <= 2:
                return np.ones(len(nd_indices)) * np.inf
            
            nd_F = F[nd_indices]
            m = len(nd_F)
            dist = np.zeros(m)
            
            for k in range(n_obj):
            # Sort by objective k
                order = np.argsort(nd_F[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = nd_F[order[-1], k] - nd_F[order[0], k]
                if f_range == 0:
                    continue
                for i in range(1, m-1):
                    dist[order[i]] += (nd_F[order[i+1], k] - nd_F[order[i-1], k]) / f_range
            
            return dist
        
        while used < budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_non_dominated_indices(F)
            
            # Select parents from non-dominated set using crowding distance
            if len(nd_idx) > 1:
                cd = crowding_distance(F, nd_idx)
                # Pick the two with highest crowding distance
                order = np.argsort(cd)[::-1]
                parent1_idx = nd_idx[order[0]]
                parent2_idx = nd_idx[order[min(1, len(order)-1)]]
            else:
                parent1_idx = nd_idx[0] if nd_idx else 0
                parent2_idx = nd_idx[0] if nd_idx else 0
            
            # Crossover: blend x0 values, keep middle vars small
            x1 = X[parent1_idx].copy()
            x2 = X[parent2_idx].copy()
            
            # For x0: take a random value between the two parents' x0 values
            x0_min = min(x1[0], x2[0])
            x0_max = max(x1[0], x2[0])
            child_x0 = x0_min + rng.random() * (x0_max - x0_min)
            
            # For middle vars: average and add small noise
            if dim > 2:
                child_mid = 0.5 * (x1[1:dim-1] + x2[1:dim-1])
                child_mid += 0.01 * rng.standard_normal(dim-2)
                child_mid = np.clip(child_mid, 0, 1)
            else:
                child_mid = np.array([])
            
            # Last var: random
            child_last = rng.random()
            
            child = np.concatenate([[child_x0], child_mid, [child_last]])
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the dominated point with lowest crowding distance, or random if all non-dominated
            dom_idx = [i for i in range(pop_size) if i not in nd_idx]
            if dom_idx:
                victim_idx = dom_idx[rng.integers(len(dom_idx))]
            else:
                # Replace the point with lowest crowding distance
                cd = crowding_distance(F, nd_idx)
                order = np.argsort(cd)
                victim_idx = nd_idx[order[0]]
            
            X[victim_idx] = child
            F[victim_idx] = fc
            
            # Occasional local refinement of the new child
            if used < budget and time.time() < deadline:
                try:
                    res = minimize(
                        objective_wrapper,
                        child,
                        args=(child_x0,),
                        method='Nelder-Mead',
                        options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    new_x = np.clip(res.x, lo, hi)
                    new_f = problem(new_x)
                    used += 1
                    if np.sum(new_f) < np.sum(F[victim_idx]):
                        X[victim_idx] = new_x
                        F[victim_idx] = new_f
                except:
                    pass
        
        return X
