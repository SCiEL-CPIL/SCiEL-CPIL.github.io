import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Phase 1: Latin Hypercube Sampling for initial spread
        N_init = min(40, budget)
        X_init = np.zeros((N_init, dim))
        for d in range(dim):
            intervals = np.linspace(lo[d], hi[d], N_init + 1)
            for i in range(N_init):
                X_init[i, d] = np.random.uniform(intervals[i], intervals[i+1])
        
        F_init = np.array([problem(x) for x in X_init])
        used = N_init
        
        # Helper: Non-dominated sort (vectorized for speed)
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            # Count how many points dominate each point
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        domination_count[j] += 1
                        dominated_set[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        domination_count[i] += 1
                        dominated_set[j].append(i)
            
            # First front
            front = [i for i in range(n) if domination_count[i] == 0]
            fronts = [front]
            
            while front:
                next_front = []
                for i in front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        # Get initial non-dominated set
        fronts_init = non_dominated_sort(F_init)
        nd_indices = fronts_init[0]
        X_nd = X_init[nd_indices]
        F_nd = F_init[nd_indices]
        
        # Phase 2: Adaptive refinement around non-dominated points
        # Use remaining budget for local search around each ND point
        remaining_budget = budget - used
        
        if remaining_budget > 0:
            n_nd = len(X_nd)
            if n_nd > 0:
                # Allocate evaluations to each ND point
                # Use more evaluations for points that are more "central" or have more neighbors
                # For simplicity, distribute evenly with some randomness
                evals_per_point = max(1, remaining_budget // n_nd)
                
                # Local search parameters
                base_radius = 0.1  # Start with 10% of span
                n_local = min(evals_per_point, 50)  # Cap per point
                
                for idx in range(n_nd):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    center = X_nd[idx]
                    # Adaptive radius: shrink as we go
                    # Use geometric decay
                    radius = base_radius * span
                    
                    for step in range(n_local):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Generate candidate near center
                        # Use Gaussian perturbation
                        perturbation = np.random.randn(dim) * radius * 0.1
                        candidate = center + perturbation
                        candidate = np.clip(candidate, lo, hi)
                        
                        f_cand = problem(candidate)
                        used += 1
                        
                        # Check if candidate is non-dominated with respect to current ND set
                        # Simple check: is it dominated by any existing ND point?
                        is_dominated = False
                        for i in range(len(F_nd)):
                            if np.all(F_nd[i] <= f_cand) and np.any(F_nd[i] < f_cand):
                                is_dominated = True
                                break
                        
                        if not is_dominated:
                            # Add to ND set and remove any points it dominates
                            new_F_nd = [f_cand]
                            new_X_nd = [candidate]
                            for i in range(len(F_nd)):
                                if not (np.all(f_cand <= F_nd[i]) and np.any(f_cand < F_nd[i])):
                                    new_F_nd.append(F_nd[i])
                                    new_X_nd.append(X_nd[i])
                            F_nd = np.array(new_F_nd)
                            X_nd = np.array(new_X_nd)
                        
                        # Shrink radius for next step
                        radius *= 0.7
        
        # Phase 3: Final polish - use remaining budget for high-precision local optimization
        # on the current best ND points using a simple gradient-free method
        remaining = budget - used
        if remaining > 0 and len(X_nd) > 0:
            # Select top few points based on hypervolume contribution
            # For simplicity, just pick the first few
            n_polish = min(3, len(X_nd))
            evals_per_polish = remaining // n_polish
            
            for i in range(n_polish):
                if used >= budget or time.time() > deadline:
                    break
                
                center = X_nd[i]
                f_center = F_nd[i]
                
                # Simple coordinate descent-like polish
                for d in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Try small steps in each direction
                    step_size = span[d] * 0.01
                    for direction in [-1, 1]:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        test = center.copy()
                        test[d] += direction * step_size
                        test = np.clip(test, lo, hi)
                        
                        f_test = problem(test)
                        used += 1
                        
                        # If test is non-dominated, move there
                        is_dominated = False
                        for j in range(len(F_nd)):
                            if j != i and np.all(F_nd[j] <= f_test) and np.any(F_nd[j] < f_test):
                                is_dominated = True
                                break
                        
                        if not is_dominated:
                            center = test
                            f_center = f_test
                            # Update ND set
                            F_nd[i] = f_center
                            X_nd[i] = center
                            # Remove any points dominated by this improved point
                            new_F_nd = [f_center]
                            new_X_nd = [center]
                            for j in range(len(F_nd)):
                                if j != i and not (np.all(f_center <= F_nd[j]) and np.any(f_center < F_nd[j])):
                                    new_F_nd.append(F_nd[j])
                                    new_X_nd.append(X_nd[j])
                            F_nd = np.array(new_F_nd)
                            X_nd = np.array(new_X_nd)
                            break  # Move to next dimension
        
        return X_nd
