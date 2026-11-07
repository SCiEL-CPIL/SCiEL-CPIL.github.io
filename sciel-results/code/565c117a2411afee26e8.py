import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer combining stratified initialization,
    Hypervolume-guided selection, and Nelder-Mead local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    return None
                all_X.append(x.copy())
                all_F.append(f)
                eval_count += 1
                return f
            except:
                return None

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        def hv_contribution_2d(F_nd, ref):
            """
            Calculate HV contribution for each point in a non-dominated set (2D).
            F_nd: (n, 2) array of non-dominated points.
            Returns: (n,) array of contributions.
            """
            if len(F_nd) == 0:
                return np.array([])
            
            # Filter points strictly better than ref
            valid_mask = (F_nd[:, 0] < ref[0]) & (F_nd[:, 1] < ref[1])
            F_valid = F_nd[valid_mask]
            
            if len(F_valid) == 0:
                return np.zeros(len(F_nd))
            
            # Sort by f1 ascending
            sort_idx = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[sort_idx]
            
            n = len(F_sorted)
            contribs = np.zeros(n)
            
            for i in range(n):
                curr_f1 = F_sorted[i, 0]
                curr_f2 = F_sorted[i, 1]
                
                if i < n - 1:
                    next_f1 = F_sorted[i+1, 0]
                else:
                    next_f1 = ref[0]
                
                width = next_f1 - curr_f1
                height = ref[1] - curr_f2
                
                contribs[i] = width * height
            
            # Map back to original indices
            full_contribs = np.zeros(len(F_nd))
            # We need to map valid indices back to original
            valid_indices = np.where(valid_mask)[0]
            full_contribs[valid_indices[sort_idx]] = contribs
            
            return full_contribs

        def nelder_mead(start_x, max_evals=10):
            """
            Perform Nelder-Mead optimization starting from start_x.
            Minimizes the scalarized objective (sum of normalized objectives)
            or simply uses the HV contribution potential.
            Here we minimize the sum of objectives to drive towards the front.
            """
            if time.time() > deadline or eval_count >= self.budget:
                return

            # Initialize simplex
            simplex = [start_x.copy()]
            for i in range(self.dim):
                delta = bounds_range[i] * 0.1
                x_new = start_x.copy()
                x_new[i] += delta
                simplex.append(np.clip(x_new, lo, hi))
            
            # Evaluate initial simplex
            f_simplex = []
            for x in simplex:
                f = evaluate(x)
                if f is None:
                    return
                f_simplex.append(f)
            
            f_simplex = np.array(f_simplex)
            
            # Main loop
            while eval_count < self.budget and time.time() < deadline and len(f_simplex) > 0:
                # Sort by sum of objectives (scalarization)
                sums = np.sum(f_simplex, axis=1)
                sort_idx = np.argsort(sums)
                
                if len(sort_idx) == 0:
                    break
                
                # Reorder simplex
                simplex = [simplex[i].copy() for i in sort_idx]
                f_simplex = f_simplex[sort_idx]
                
                worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Check convergence or budget
                if eval_count >= self.budget or time.time() > deadline:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = evaluate(x_ref)
                if f_ref is None:
                    break
                
                if np.sum(f_ref) < np.sum(f_simplex[0]):
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if f_exp is None:
                        break
                    
                    if np.sum(f_exp) < np.sum(f_ref):
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif np.sum(f_ref) < np.sum(f_simplex[-2]):
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    beta = 0.5
                    x_con = centroid + beta * (worst - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = evaluate(x_con)
                    if f_con is None:
                        break
                    
                    if np.sum(f_con) < np.sum(f_worst):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = evaluate(simplex[i])
                            if f_simplex[i] is None:
                                return

        # Phase 1: Stratified Initialization
        # Use a grid of 3^4 = 81 points for good coverage
        n_grid = 81
        grid_points = []
        # 3 levels per dimension: 0, 0.5, 1.0
        levels = [0.0, 0.5, 1.0]
        for l0 in levels:
            for l1 in levels:
                for l2 in levels:
                    for l3 in levels:
                        x = lo + bounds_range * np.array([l0, l1, l2, l3])
                        grid_points.append(x)
        
        # Add some random points to fill up to ~100
        n_random = 20
        random_points = lo + bounds_range * np.random.random((n_random, self.dim))
        
        initial_points = np.vstack([grid_points, random_points])
        
        for x in initial_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Iterative Refinement
        # Select top points based on HV contribution and run Nelder-Mead on them
        
        while eval_count < self.budget - 10 and time.time() < deadline:
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            nd_F = np.array([all_F[i] for i in nd_indices])
            
            # Calculate HV contributions
            contribs = hv_contribution_2d(nd_F, ref_point)
            
            # Select top 5 points with highest contribution
            # If all contributions are 0, select by sum of objectives
            if np.sum(contribs) < 1e-10:
                sums = np.sum(nd_F, axis=1)
                top_idx = np.argsort(sums)[:5]
            else:
                top_idx = np.argsort(contribs)[::-1][:5]
            
            selected_indices = nd_indices[top_idx]
            
            # Run Nelder-Mead on each selected point
            for idx in selected_indices:
                if time.time() > deadline or eval_count >= self.budget:
                    break
                
                start_x = all_X[idx].copy()
                # Allocate up to 10 evaluations per local search
                # But ensure we don't exceed budget
                remaining = self.budget - eval_count
                if remaining < 5:
                    break
                    
                nelder_mead(start_x, max_evals=min(10, remaining))

        return all_X
