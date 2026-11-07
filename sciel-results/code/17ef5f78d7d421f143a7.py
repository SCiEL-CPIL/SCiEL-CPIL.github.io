"""Multi-objective optimizer on RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a hybrid of random sampling, local search,
    and NSGA-II-style evolutionary steps with a time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # We will track all evaluated points and their objective values
        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            X_list.append(x)
            F_list.append(f)
            eval_count += 1
            return f

        # Check time budget
        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # Phase 1: Dense random sampling to get a good initial spread
        # Use a significant portion of budget for random sampling
        n_random = min(budget // 2, 100)
        for i in range(n_random):
            if not time_ok():
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # Phase 2: Local search around best points
        # Find non-dominated points from random sampling
        F_arr = np.array(F_list) if F_list else np.empty((0, n_obj))
        if len(F_list) > 0:
            # Find non-dominated indices
            nondom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom_mask[i] = False
                        break
            nondom_indices = np.where(nondom_mask)[0]
            if len(nondom_indices) > 0:
                # Local search around each non-dominated point
                max_local = min(budget - eval_count, 80)
                for idx in nondom_indices[:min(len(nondom_indices), 10)]:
                    if not time_ok():
                        break
                    base_x = X_list[idx].copy()
                    base_f = F_list[idx].copy()
                    # Try perturbations with decreasing step sizes
                    for step in [0.1, 0.05, 0.02, 0.01]:
                        if not time_ok():
                            break
                        for _ in range(4):
                            if not time_ok():
                                break
                            perturbation = np.random.randn(dim) * step * (hi - lo)
                            new_x = base_x + perturbation
                            new_f = evaluate(new_x)
                            # If better, update base
                            if np.all(new_f <= base_f) and np.any(new_f < base_f):
                                base_x = new_x.copy()
                                base_f = new_f.copy()

        # Phase 3: NSGA-II style evolutionary step
        if time_ok() and eval_count < budget:
            # Build population from all evaluated points, keeping non-dominated
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)
            
            # Get non-dominated set
            nondom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom_mask[i] = False
                        break
            nondom_idx = np.where(nondom_mask)[0]
            
            if len(nondom_idx) >= 2:
                parents_x = X_arr[nondom_idx]
                parents_f = F_arr[nondom_idx]
                
                # Evolve for remaining budget
                n_evolve = min(budget - eval_count, 60)
                for _ in range(n_evolve):
                    if not time_ok():
                        break
                    # Select two parents via tournament
                    i1 = np.random.randint(len(parents_x))
                    i2 = np.random.randint(len(parents_x))
                    # Simple crossover
                    alpha = np.random.random(dim)
                    child = alpha * parents_x[i1] + (1 - alpha) * parents_x[i2]
                    # Mutation
                    child += 0.05 * (hi - lo) * np.random.randn(dim)
                    child = np.clip(child, lo, hi)
                    
                    child_f = evaluate(child)
                    
                    # Add to parent pool if non-dominated
                    is_dom = False
                    for pf in parents_f:
                        if np.all(pf <= child_f) and np.any(pf < child_f):
                            is_dom = True
                            break
                    if not is_dom:
                        parents_x = np.vstack([parents_x, child])
                        parents_f = np.vstack([parents_f, child_f])
                        # Remove dominated points from parent pool
                        keep = np.ones(len(parents_f), dtype=bool)
                        for k in range(len(parents_f)):
                            for m in range(len(parents_f)):
                                if k == m:
                                    continue
                                if np.all(parents_f[m] <= parents_f[k]) and np.any(parents_f[m] < parents_f[k]):
                                    keep[k] = False
                                    break
                        parents_x = parents_x[keep]
                        parents_f = parents_f[keep]

        return X_list
# EVOLVE-BLOCK-END
