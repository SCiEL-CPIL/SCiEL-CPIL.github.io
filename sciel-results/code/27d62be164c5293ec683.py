import numpy as np
import time

class Optimizer:
            def __init__(self, budget, dim, n_obj):
                self.budget = budget
                self.dim = dim
                self.n_obj = n_obj

            def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 290 # Safety margin
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                span = hi - lo
                
                # Use global np.random
                # Evaluator seeds it.
                
                pop_size = 20
                # Initialize
                X = lo + span * np.random.random((pop_size, self.dim))
                F = np.array([problem(x) for x in X])
                used = pop_size
                
                # Helper: Non-dominated sort / filter
                def get_non_dominated(F):
                    n = len(F)
                    dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        if dominated[i]: continue
                        for j in range(n):
                            if i == j or dominated[j]: continue
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated[i] = True
                                break
                    return np.where(~dominated)[0]
                
                def crowding_distance(F_nd):
                    # F_nd is array of non-dominated points
                    m = len(F_nd)
                    if m <= 2:
                        return np.full(m, np.inf)
                    cd = np.zeros(m)
                    for k in range(self.n_obj):
                        idx = np.argsort(F_nd[:, k])
                        cd[idx[0]] = np.inf
                        cd[idx[-1]] = np.inf
                        for i in range(1, m-1):
                            cd[idx[i]] += (F_nd[idx[i+1], k] - F_nd[idx[i-1], k]) / (F_nd[idx[-1], k] - F_nd[idx[0], k] + 1e-9)
                    return cd
                
                sigma = 0.5
                while used < self.budget:
                    if time.time() > deadline:
                        break
                    
                    # Get non-dominated indices
                    nd_idx = get_non_dominated(F)
                    if len(nd_idx) == 0:
                        nd_idx = np.arange(len(F))
                    
                    # Select parent
                    # Prefer non-dominated. If none, pick random.
                    # Use uniform selection from non-dominated.
                    parent_idx = nd_idx[np.random.randint(len(nd_idx))]
                    parent = X[parent_idx]
                    
                    # Mutate
                    child = parent + sigma * span * np.random.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # Update population
                    # Simple strategy: Replace worst non-dominated point or dominated point?
                    # Let's try: If child is non-dominated w.r.t X, add it.
                    # If pop grows, trim.
                    
                    # Check if child is dominated by any in X
                    is_dominated = False
                    for i in range(len(X)):
                        if np.all(F[i] <= fc) and np.any(F[i] < fc):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        # Add child
                        X = np.vstack([X, child])
                        F = np.vstack([F, fc])
                        
                        # Trim if too large
                        if len(X) > pop_size + 5: # Allow some growth
                            # Calculate crowding distance for all non-dominated
                            nd_idx = get_non_dominated(F)
                            if len(nd_idx) > pop_size:
                                F_nd = F[nd_idx]
                                cd = crowding_distance(F_nd)
                                # Find index in nd_idx with lowest cd
                                # But we need to remove from X/F
                                # Sort nd_idx by cd
                                order = np.argsort(cd)
                                # Remove the one with lowest cd
                                worst_in_nd = nd_idx[order[0]]
                                # Remove worst_in_nd from X and F
                                mask = np.ones(len(X), dtype=bool)
                                mask[worst_in_nd] = False
                                X = X[mask]
                                F = F[mask]
                    else:
                        # Child is dominated.
                        # Maybe replace the worst point in the population?
                        # Or discard?
                        # Replacing the worst (max sum) is what the current code does.
                        # Let's try replacing the point with the highest sum of objectives.
                        worst_idx = np.argmax(np.sum(F, axis=1))
                        X[worst_idx] = child
                        F[worst_idx] = fc
                    
                    # Decay sigma
                    sigma *= 0.98
                
                return X
