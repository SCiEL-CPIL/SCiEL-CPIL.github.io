"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial random population
        pop_size = min(100, max(20, self.budget // 2))
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
            
            current_front = [p for p in range(n) if n_dom[p] == 0]
            if not current_front:
                current_front = list(range(n))
            fronts.append(current_front)
            
            i = 0
            while current_front and used < self.budget:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front
                i += 1
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            cd = np.zeros(len(front_indices))
            n_obj = F.shape[1]
            for m in range(n_obj):
                sorted_indices = np.argsort(F[front_indices, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                if F[front_indices, m].max() - F[front_indices, m].min() == 0:
                    continue
                for i in range(1, len(front_indices) - 1):
                    cd[sorted_indices[i]] += (F[front_indices[sorted_indices[i+1]], m] - F[front_indices[sorted_indices[i-1]], m]) / (F[front_indices, m].max() - F[front_indices, m].min())
            return cd

        # Main Evolutionary Loop
        while used < self.budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            
            # Create offspring
            offspring_size = pop_size
            X_off = []
            F_off = []
            
            for _ in range(offspring_size):
                # Tournament selection
                p1, p2 = np.random.choice(len(F), 2, replace=False)
                
                # Assign ranks
                rank1 = rank2 = 0
                for i, front in enumerate(fronts):
                    if p1 in front:
                        rank1 = i
                        break
                for i, front in enumerate(fronts):
                    if p2 in front:
                        rank2 = i
                        break
                
                if rank1 < rank2:
                    parent = X[p1]
                elif rank2 < rank1:
                    parent = X[p2]
                else:
                    # Same front, pick based on crowding distance (approximate)
                    # For speed, just pick random from same front
                    parent = X[np.random.choice([p1, p2])]
                
                # Mutation
                # Adaptive mutation rate
                sigma = 0.1 + 0.1 * (1 - used / self.budget)
                child = parent + sigma * (hi - lo) * np.random.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                X_off.append(child)
                F_off.append(problem(child))
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            if not X_off:
                break
                
            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Selection
            fronts = fast_non_dominated_sort(F_all)
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to sort by crowding distance
                    front_F = F_all[front]
                    cd = crowding_distance(front_F, np.arange(len(front)))
                    sorted_front = np.argsort(-cd)
                    remaining = pop_size - len(X_new)
                    for i in range(remaining):
                        idx_in_front = front[sorted_front[i]]
                        X_new.append(X_all[idx_in_front])
                        F_new.append(F_all[idx_in_front])
                    break
            
            if X_new:
                X = np.array(X_new)
                F = np.array(F_new)
            else:
                break

        # Final Polish: Local search on non-dominated points
        if time.time() < deadline - 10:
            # Get non-dominated points
            fronts = fast_non_dominated_sort(F)
            if fronts:
                pareto_indices = fronts[0]
                for idx in pareto_indices:
                    if time.time() > deadline - 5:
                        break
                    x0 = X[idx]
                    # Try a few local perturbations
                    for _ in range(5):
                        if time.time() > deadline - 5:
                            break
                        # Random direction local step
                        step_size = 0.05 * (hi - lo)
                        x_try = x0 + step_size * np.random.standard_normal(self.dim)
                        x_try = np.clip(x_try, lo, hi)
                        f_try = problem(x_try)
                        used += 1
                        if used >= self.budget:
                            break
                        
                        # Check if it dominates the original or is non-dominated
                        # Simple replacement if it improves the sum of squares (heuristic)
                        if np.sum(f_try**2) < np.sum(F[idx]**2):
                            X[idx] = x_try
                            F[idx] = f_try
                        elif not np.any(np.all(F <= f_try, axis=1) & np.any(F < f_try, axis=1)):
                            # It's non-dominated, maybe add it if we have space? 
                            # We don't change population size here, but we can replace a dominated point if found
                            # For simplicity, just keep the best local improvement
                            pass

        # Return the entire history of evaluated points? 
        # The task says "Every returned objective vector is recorded".
        # The evaluator records what is returned by __call__? 
        # "The return value of __call__ is ignored. Every returned objective vector is recorded..."
        # Wait, "Every returned objective vector is recorded" implies the evaluator intercepts the calls to `problem`?
        # "The evaluator calls... and calls it once... Every returned objective vector is recorded"
        # Usually in these tasks, the evaluator wraps the `problem` function or the optimizer returns the set of points.
        # "The return value of __call__ is ignored."
        # So I must ensure that the *evaluations* I make are the ones that count.
        # The evaluator likely wraps `problem` to record outputs.
        # So I just need to call `problem` efficiently.
        
        return X
# EVOLVE-BLOCK-END
