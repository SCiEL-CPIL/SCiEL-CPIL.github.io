"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using global initialization and local Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 320.0  # Safety margin

    def __call__(self, problem, bounds):
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # 1. Global Initialization
        n_init = 20
        if self.budget < n_init:
            n_init = self.budget
        
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.zeros((n_init, self.n_obj))
        
        for i in range(n_init):
            if self.eval_count >= self.budget or time.time() > self.deadline:
                break
            F_init[i] = problem(X_init[i])
            self.eval_count += 1

        # Identify non-dominated points
        # A point i is dominated by j if j <= i in all objectives and j < i in at least one
        def is_dominated(f, F):
            # F is (N, n_obj), f is (n_obj,)
            # Check if any row in F dominates f
            # F <= f (all objectives) AND F < f (at least one objective)
            le = np.all(F <= f, axis=1)
            lt = np.any(F < f, axis=1)
            return np.any(le & lt)

        non_dom_indices = []
        for i in range(n_init):
            # Check against all other points
            dominated = False
            for j in range(n_init):
                if i == j:
                    continue
                if is_dominated(F_init[i], F_init[j]):
                    dominated = True
                    break
            if not dominated:
                non_dom_indices.append(i)
        
        if not non_dom_indices:
            # Fallback: take the first point
            non_dom_indices = [0]

        # 2. Local Refinement
        # We will pick up to 5 points from the non-dominated set to refine.
        # Strategy: Pick points that are extreme in each objective and one balanced.
        
        # Get the non-dominated points
        X_nd = X_init[non_dom_indices]
        F_nd = F_init[non_dom_indices]
        
        # Define targets for local search
        # 1. Min f1
        idx_min_f1 = np.argmin(F_nd[:, 0])
        # 2. Min f2
        idx_min_f2 = np.argmin(F_nd[:, 1])
        # 3. Min f3
        idx_min_f3 = np.argmin(F_nd[:, 2])
        # 4. Min sum
        sums = np.sum(F_nd, axis=1)
        idx_min_sum = np.argmin(sums)
        # 5. Min distance to ideal (min of each objective in F_nd)
        ideal = np.min(F_nd, axis=0)
        dists = np.linalg.norm(F_nd - ideal, axis=1)
        idx_min_dist = np.argmin(dists)

        # Unique indices to avoid redundant work
        target_indices = list(set([idx_min_f1, idx_min_f2, idx_min_f3, idx_min_sum, idx_min_dist]))
        
        # Allocate remaining budget
        remaining_budget = self.budget - self.eval_count
        n_targets = len(target_indices)
        if n_targets == 0:
            return None
            
        maxfev_per_target = max(1, remaining_budget // n_targets)
        
        # Define wrapper for problem to count evals
        def wrapped_problem(x):
            if self.eval_count >= self.budget or time.time() > self.deadline:
                # Return a large value to stop optimization
                return [1e6] * self.n_obj
            f = problem(x)
            self.eval_count += 1
            return f

        # Define scalar objectives for each target
        # Note: We need to be careful with the objective functions passed to minimize.
        # They should return a scalar.
        
        def make_obj_func(target_type):
            if target_type == 0: # Min f1
                def obj(x):
                    f = wrapped_problem(x)
                    return f[0]
                return obj
            elif target_type == 1: # Min f2
                def obj(x):
                    f = wrapped_problem(x)
                    return f[1]
                return obj
            elif target_type == 2: # Min f3
                def obj(x):
                    f = wrapped_problem(x)
                    return f[2]
                return obj
            elif target_type == 3: # Min sum
                def obj(x):
                    f = wrapped_problem(x)
                    return np.sum(f)
                return obj
            else: # Min dist
                def obj(x):
                    f = wrapped_problem(x)
                    return np.linalg.norm(f - ideal)
                return obj

        # Run local searches
        for k, idx in enumerate(target_indices):
            if self.eval_count >= self.budget or time.time() > self.deadline:
                break
                
            x0 = X_nd[idx]
            target_type = k # 0, 1, 2, 3, 4 corresponding to the logic above
            
            # Map k to the actual target type logic used in make_obj_func
            # The list target_indices was built from specific indices, but the *type* of objective 
            # should correspond to the strategy (Min f1, Min f2, etc).
            # My previous mapping of k to type is not 1-to-1 with the specific index chosen.
            # I should define the objective based on the *strategy*, not the index.
            
            # Let's re-structure:
            # Strategies:
            # 1. Minimize f1
            # 2. Minimize f2
            # 3. Minimize f3
            # 4. Minimize sum
            # 5. Minimize dist
            
            # We have 5 strategies. We will run one local search for each strategy, 
            # starting from the best point for that strategy in the initial population.
            
            pass # This loop structure is slightly off, let's rewrite the loop below.

        # Corrected Loop
        strategies = [
            ("min_f1", idx_min_f1),
            ("min_f2", idx_min_f2),
            ("min_f3", idx_min_f3),
            ("min_sum", idx_min_sum),
            ("min_dist", idx_min_dist)
        ]
        
        for strategy_name, idx in strategies:
            if self.eval_count >= self.budget or time.time() > self.deadline:
                break
                
            x0 = X_nd[idx]
            
            if strategy_name == "min_f1":
                def obj(x):
                    f = wrapped_problem(x)
                    return f[0]
            elif strategy_name == "min_f2":
                def obj(x):
                    f = wrapped_problem(x)
                    return f[1]
            elif strategy_name == "min_f3":
                def obj(x):
                    f = wrapped_problem(x)
                    return f[2]
            elif strategy_name == "min_sum":
                def obj(x):
                    f = wrapped_problem(x)
                    return np.sum(f)
            else: # min_dist
                def obj(x):
                    f = wrapped_problem(x)
                    return np.linalg.norm(f - ideal)

            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev_per_target, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                pass

        return None

# EVOLVE-BLOCK-END
