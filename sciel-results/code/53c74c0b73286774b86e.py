"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Improved evolutionary multi-objective search with crowding distance and dominance-based replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop = int(min(20, self.budget // 4))
        
        X = lo + (hi - lo) * np.random.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        start_time = time.time()
        deadline = start_time + 290
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Selection: Random non-dominated parent
            dominated = np.zeros(pop, dtype=bool)
            for i in range(pop):
                for j in range(pop):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
                        
            if np.any(~dominated):
                nd_idx = np.where(~dominated)[0]
                parent_idx = np.random.randint(len(nd_idx))
                parent = X[nd_idx[parent_idx]]
            else:
                parent = X[np.random.randint(pop)]
                
            # Mutation
            sigma = 0.1
            child = parent + sigma * (hi - lo) * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replacement Strategy
            replaced = False
            for i in range(pop):
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break
                    
            if not replaced:
                is_dominated = False
                for i in range(pop):
                    if np.all(F[i] <= fc) and np.any(F[i] < fc):
                        is_dominated = True
                        break
                        
                if is_dominated:
                    # Replace the most dominated point
                    dom_counts = np.zeros(pop, dtype=int)
                    for i in range(pop):
                        for j in range(pop):
                            if i == j: continue
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dom_counts[i] += 1
                    worst_idx = np.argmax(dom_counts)
                    X[worst_idx] = child
                    F[worst_idx] = fc
                else:
                    # Replace the most crowded point (min crowding distance)
                    crowding = np.zeros(pop)
                    for m in range(self.n_obj):
                        f_m = F[:, m]
                        f_min, f_max = np.min(f_m), np.max(f_m)
                        if f_max == f_min:
                            continue
                        idx = np.argsort(f_m)
                        crowding[idx[0]] = np.inf
                        crowding[idx[-1]] = np.inf
                        for k in range(1, pop - 1):
                            i_curr = idx[k]
                            i_prev = idx[k-1]
                            i_next = idx[k+1]
                            dist = (F[i_next, m] - F[i_prev, m]) / (f_max - f_min)
                            crowding[i_curr] += dist
                            
                    worst_idx = np.argmin(crowding)
                    X[worst_idx] = child
                    F[worst_idx] = fc
                    
        return X
# EVOLVE-BLOCK-END
