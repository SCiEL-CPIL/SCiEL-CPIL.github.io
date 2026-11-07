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
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        all_X = []
        all_F = []
        evals_used = 0

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            evals_used += 1
            return f

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(F_mat):
            n = len(F_mat)
            if n == 0:
                return []
            front = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and dominates(F_mat[j], F_mat[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            return front

        # Phase 1: Initialize with a mix of points
        # WFG9 has a known Pareto front. Let's try to sample near it.
        # The Pareto front of WFG9 is known to have specific structure.
        # For 3 objectives, the front is a 2D surface.
        
        # Initial random samples
        n_init = 20
        for _ in range(n_init):
            if evals_used >= max_evals:
                break
            x = np.random.uniform(lo, hi)
            evaluate(x)

        # Add points near the origin (often good for WFG)
        for _ in range(10):
            if evals_used >= max_evals:
                break
            x = lo + 0.1 * (hi - lo) * np.random.rand(self.dim)
            evaluate(x)

        # Add points along the axes
        for d in range(self.dim):
            if evals_used >= max_evals:
                break
            x = np.random.uniform(lo, hi)
            x[d] = lo[d]
            evaluate(x)
            x[d] = hi[d]
            evaluate(x)

        # Phase 2: Local optimization from promising points
        # Select a subset of non-dominated points to optimize
        if all_F:
            F_arr = np.array(all_F)
            front_indices = non_dominated_front(F_arr)
            
            # Sort by sum of objectives to pick "best" ones
            sums = np.sum(F_arr, axis=1)
            front_indices = [front_indices[i] for i in np.argsort(sums[front_indices])]
            
            # Pick up to 15 points for local search
            n_local = min(15, len(front_indices))
            for idx in front_indices[:n_local]:
                if evals_used >= max_evals - 5:
                    break
                if time.time() > deadline:
                    break
                
                x0 = all_X[idx].copy()
                
                # Define a scalarized objective using weighted sum with random weights
                # This helps explore different parts of the front
                for _ in range(3):  # 3 restarts with different weights
                    if evals_used >= max_evals - 2:
                        break
                    if time.time() > deadline:
                        break
                    
                    weights = np.random.dirichlet(np.ones(self.n_obj))
                    
                    def objective(x):
                        if evals_used >= max_evals:
                            return np.inf
                        x = np.clip(x, lo, hi)
                        f = np.asarray(problem(x), dtype=float)
                        all_X.append(x.copy())
                        all_F.append(f.copy())
                        evals_used += 1
                        return np.dot(weights, f)
                    
                    try:
                        res = minimize(objective, x0, method='L-BFGS-B', 
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 10, 'maxls': 5})
                    except Exception:
                        pass
        
        # Phase 3: If budget remains, do a few more random explorations
        while evals_used < max_evals:
            if time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            evaluate(x)

        # Return all evaluated points (the evaluator records all objective vectors returned)
        # The return value is ignored, but we return something
        return all_X
