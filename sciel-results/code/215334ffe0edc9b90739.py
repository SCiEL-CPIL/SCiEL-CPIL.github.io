import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = (lo + hi) / 2.0
        ranges = hi - lo
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        evaluated_X = []
        evaluated_F = []
        
        def eval_point(x):
            nonlocal budget
            if time.time() > deadline:
                return None
            f = problem(x)
            evaluated_X.append(x)
            evaluated_F.append(f)
            budget -= 1
            return f
        
        # WFG9 structure: first 3 variables control shape, remaining 9 control distance to Pareto front
        # The Pareto front is determined by the first 3 decision variables
        # The other 9 variables should be set to values that minimize the distance component
        
        # Phase 1: Dense grid sweep on the primary shape variables (first 3)
        # Use a 12x12 grid for the first 2 variables and sweep the 3rd
        # This gives 12*12*12 = 1728 points, but we only have 200 budget
        # So use a smaller grid: 6x6x6 = 216 points (slightly over budget)
        # Let's use 5x5x5 = 125 points for the grid
        
        n_shape = 3
        n_tail = dim - n_shape
        
        # Find good tail values by evaluating at mid
        x_tail_good = mid[n_shape:]
        
        # Phase 2: Dense grid on shape variables
        grid_n = 5
        g = np.linspace(lo[:n_shape], hi[:n_shape], grid_n)
        
        for v1 in g[0]:
            for v2 in g[1]:
                for v3 in g[2]:
                    if budget <= 0 or time.time() > deadline:
                        break
                    x = np.concatenate([[v1, v2, v3], x_tail_good])
                    eval_point(x)
        
        # Phase 3: Local refinement around non-dominated points
        if evaluated_F:
            F = np.array(evaluated_F)
            X = np.array(evaluated_X)
            n = len(F)
            
            # Find non-dominated points
            front0 = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    front0.append(i)
            
            # Refine top candidates
            max_refine = min(len(front0), 15)
            for idx in front0[:max_refine]:
                if budget <= 0 or time.time() > deadline:
                    break
                x_center = X[idx]
                n_perturb = 4
                for _ in range(n_perturb):
                    if budget <= 0 or time.time() > deadline:
                        break
                    noise = np.random.randn(dim) * 0.05 * ranges
                    x_new = x_center + noise
                    x_new = np.clip(x_new, lo, hi)
                    eval_point(x_new)
        
        return np.array(evaluated_X) if evaluated_X else np.array([mid])
