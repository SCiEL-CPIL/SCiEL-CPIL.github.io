"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining random sampling, gradient-based local search,
    and a simple evolutionary strategy with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Store all evaluated points and their objective vectors
        X = []  # list of x vectors
        F = []  # list of f vectors
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        # Phase 1: Random sampling (30% of budget)
        n_random = max(10, self.budget // 3)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            eval_point(x)

        # Phase 2: Local search from best random points
        # Pick a few diverse points and run local optimization
        if F and time.time() < deadline:
            # Sort by sum of objectives (rough indicator)
            F_arr = np.array(F)
            # Filter out points dominated by reference [1705, 12, 0.3]
            ref = np.array([1705.0, 12.0, 0.3])
            valid_mask = np.all(F_arr < ref, axis=1)
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                # Sort by sum
                sums = F_arr[valid_mask].sum(axis=1)
                sorted_idx = valid_idx[np.argsort(sums)]
                
                # Take top 5 diverse points for local search
                n_local = min(5, len(sorted_idx), max(1, (self.budget - used) // 10))
                for idx in sorted_idx[:n_local]:
                    if time.time() > deadline or used >= self.budget:
                        break
                    x0 = X[idx]
                    
                    # Try optimizing each objective separately
                    for obj_i in range(self.n_obj):
                        if time.time() > deadline or used >= self.budget:
                            break
                        
                        def objective(x, _obj_i=obj_i):
                            nonlocal used
                            if used >= self.budget:
                                return 0.0
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x))
                            X.append(x)
                            F.append(f)
                            used += 1
                            return f[_obj_i]
                        
                        try:
                            res = minimize(objective, x0, method='L-BFGS-B',
                                         bounds=list(zip(lo, hi)),
                                         options={'maxiter': 20, 'maxfun': 30})
                        except Exception:
                            pass

        # Phase 3: Evolutionary strategy with the remaining budget
        if used < self.budget and time.time() < deadline:
            # Build population from non-dominated points
            F_arr = np.array(F)
            
            def non_dominated_sort(F):
                """Return indices of non-dominated points."""
                n = len(F)
                if n == 0:
                    return []
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    if dominated[i]:
                        continue
                    for j in range(n):
                        if i == j or dominated[j]:
                            continue
                        # Check if j dominates i
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                        # Check if i dominates j
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            dominated[j] = True
                return [i for i in range(n) if not dominated[i]]
            
            # Main evolutionary loop
            pop_size = max(10, min(30, (self.budget - used) // 5))
            
            # Initialize population from non-dominated points
            nd_idx = non_dominated_sort(F_arr)
            if len(nd_idx) < pop_size:
                # Add random points to fill
                while len(nd_idx) < pop_size and used < self.budget:
                    x = lo + (hi - lo) * rng.random(self.dim)
                    f = eval_point(x)
                    if f is not None:
                        nd_idx.append(used - 1)
            
            # Take first pop_size non-dominated points
            pop_idx = nd_idx[:pop_size]
            
            while used < self.budget and time.time() < deadline:
                # Select parents from non-dominated set
                F_arr = np.array(F)
                nd_idx = non_dominated_sort(F_arr)
                if len(nd_idx) < 2:
                    nd_idx = list(range(len(F_arr)))
                
                # Pick two parents
                p1 = nd_idx[rng.integers(len(nd_idx))]
                p2 = nd_idx[rng.integers(len(nd_idx))]
                
                # Crossover
                x1 = X[p1]
                x2 = X[p2]
                alpha = rng.random(self.dim)
                child = alpha * x1 + (1 - alpha) * x2
                child = np.clip(child, lo, hi)
                
                # Mutation
                mutation_rate = 0.3
                if rng.random() < mutation_rate:
                    child += 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                
                fc = eval_point(child)
                if fc is None:
                    break

        return X
# EVOLVE-BLOCK-END
