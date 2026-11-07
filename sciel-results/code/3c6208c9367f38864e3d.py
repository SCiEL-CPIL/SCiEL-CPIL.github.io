"""Multi-objective optimizer for RE34: Sobol sampling + weighted-sum local search."""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables).
    Strategy:
    1. Phase 1 (120 evals): Sobol quasi-random sampling for broad coverage.
    2. Phase 2 (80 evals): Weighted-sum scalarization local search from diverse
       starting points on the non-dominated front, using multiple weight vectors
       to explore different regions of the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        eval_count = 0
        max_evals = self.budget

        def eval_problem(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return np.array(problem(x))

        X_list = []
        F_list = []

        # Phase 1: Sobol quasi-random initial sampling
        n_sobol = min(120, max_evals)
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            samples = sampler.random(n_sobol)
        except Exception:
            samples = np.random.rand(n_sobol, self.dim)

        X_init = lo + samples * span

        for x in X_init:
            if eval_count >= max_evals:
                break
            f = eval_problem(x)
            X_list.append(x.copy())
            F_list.append(f.copy())

        # Helper: get non-dominated indices
        def get_non_dominated_indices(F_arr):
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Phase 2: Weighted-sum local search from diverse starting points
        n_refine = max_evals - eval_count
        if n_refine > 0 and time.time() < deadline:
            F_arr = np.array(F_list)
            nd_indices = get_non_dominated_indices(F_arr)

            if len(nd_indices) > 0:
                # Select diverse starting points from the non-dominated front
                # Use k-means-like selection: pick points that are far apart in objective space
                nd_points = F_arr[nd_indices]
                
                # Normalize for distance computation
                f_min = nd_points.min(axis=0)
                f_max = nd_points.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                nd_norm = (nd_points - f_min) / f_range

                # Select up to 10 diverse starting points
                n_starts = min(10, len(nd_indices))
                if n_starts == 1:
                    start_indices = [nd_indices[0]]
                else:
                    # Farthest-point sampling in normalized objective space
                    start_indices = [nd_indices[np.random.randint(len(nd_indices))]]
                    for _ in range(1 - n_starts):
                        pass
                    # Proper farthest-point sampling
                    selected = [np.random.randint(len(nd_indices))]
                    while len(selected) < n_starts:
                        # For each candidate, compute min distance to selected
                        best_idx = -1
                        best_dist = -1
                        for i in range(len(nd_indices)):
                            if i in selected:
                                continue
                            min_dist = min(
                                np.linalg.norm(nd_norm[i] - nd_norm[s]) 
                                for s in selected
                            )
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_idx = i
                        if best_idx >= 0:
                            selected.append(best_idx)
                        else:
                            break
                    start_indices = [nd_indices[i] for i in selected]

                # Define weight vectors for different regions of the Pareto front
                # For 3 objectives, use weight vectors that emphasize different trade-offs
                weight_vectors = [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.1, 0.7],
                    [0.4, 0.4, 0.2],
                    [0.4, 0.2, 0.4],
                    [0.2, 0.4, 0.4],
                    [0.33, 0.33, 0.34],
                    [0.5, 0.25, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.25, 0.5],
                ]
                weight_vectors = np.array(weight_vectors)

                # For each starting point and weight vector, run a local search
                evals_per_start = max(3, n_refine // len(start_indices))
                
                for start_idx in start_indices:
                    if eval_count >= max_evals:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_start = X_list[start_idx].copy()
                    f_start = F_list[start_idx].copy()
                    
                    # Try different weight vectors from this starting point
                    # Pick 2-3 weight vectors that are "far" from the current point's implied weights
                    for w in weight_vectors:
                        if eval_count >= max_evals:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Weighted-sum objective
                        def weighted_obj(x):
                            f = problem(x)
                            return np.dot(w, f)
                        
                        # Run local search
                        try:
                            res = minimize(
                                weighted_obj,
                                x_start,
                                bounds=list(zip(lo, hi)),
                                method='L-BFGS-B',
                                options={'maxiter': 5, 'ftol': 1e-8, 'gtol': 1e-5}
                            )
                            if res.success or res.fun < np.dot(w, f_start):
                                x_opt = res.x
                                f_opt = eval_problem(x_opt)
                                X_list.append(x_opt.copy())
                                F_list.append(f_opt.copy())
                        except Exception:
                            pass

        return X_list
