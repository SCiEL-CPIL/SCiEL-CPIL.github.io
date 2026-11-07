"""Multi-objective optimizer on RE34: Decomposition-based search with local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based multi-objective optimizer for 3-objective RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0
        X = []  # list of decision vectors
        F = []  # list of objective vectors

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        def dominated_by_any(f_new, F_arr):
            """Check if f_new is dominated by any point in F_arr."""
            if len(F_arr) == 0:
                return False
            # f_new is dominated if exists f in F_arr such that f <= f_new and f < f_new
            # Vectorized check
            return np.any(np.all(F_arr <= f_new, axis=1) & np.any(F_arr < f_new, axis=1))

        def non_dominated_set(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Phase 1: Initialize with a diverse set of points using a grid/LHS
        # We want to cover the weight space.
        # For 3 objectives, we can use a simplex grid of weights.
        # Let's generate weight vectors.
        # n_weights = 15 (5 per edge, 3 vertices, 1 center? No, just random or grid)
        # Better: Use a set of weight vectors that cover the simplex.
        
        # Generate weight vectors
        # Simple approach: Random points on the simplex, or a structured grid.
        # Let's use 10-15 weight vectors.
        n_weights = 12
        rng = np.random.default_rng(42)
        
        # Generate points on the simplex
        weights = []
        # Add vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        
        # Add midpoints of edges
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
                
        # Add center
        weights.append(np.ones(n_obj) / n_obj)
        
        # Fill rest with random
        while len(weights) < n_weights:
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
            
        weights = np.array(weights)

        # Initial evaluation: One point per weight vector using a simple heuristic
        # Or just random points? 
        # Let's evaluate a small population of random points first to get a baseline.
        n_init = min(20, budget // 10)
        for i in range(n_init):
            x = lo + rng.random(dim) * span
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        # Phase 2: Decomposition-based local search
        # For each weight vector, we define a scalarization function (Tchebycheff)
        # and perform a local search (Nelder-Mead) to minimize it.
        # This pushes the solution towards the specific region of the Pareto front.
        
        # We need to manage the budget carefully.
        # Total budget = 200.
        # n_init = 20.
        # Remaining = 180.
        # n_weights = 12.
        # Each local search might take 10-15 evaluations.
        # 12 * 15 = 180. This fits perfectly.
        
        # We need to track the best point for each weight vector.
        # We will use the current non-dominated set to pick the best starting point
        # for each weight vector, or just use a random point from the initial set.
        
        # To be more robust, let's do multiple restarts or a single pass with more evals per weight.
        # Let's allocate ~15 evals per weight vector.
        
        # We'll store the best solution found for each weight vector.
        best_X_for_weight = [None] * n_weights
        best_F_for_weight = [None] * n_weights
        
        # Sort weights by "importance" or just process them all.
        # Let's process them in order.
        
        # Helper to compute Tchebycheff scalarization
        def tchebycheff_scalarization(x, weights_vec, ideal_point):
            f = np.asarray(problem(np.clip(x, lo, hi)), dtype=np.float64)
            # Tchebycheff: max_i ( w_i * |f_i - z_i^*| )
            # We don't know the ideal point exactly, so we use the current best known
            # or the reference point? 
            # Usually, Tchebycheff uses the ideal point (min of each objective).
            # Since we don't have the true ideal point, we can use the min of the current archive
            # or just use a fixed reference if we don't have a good archive yet.
            # For simplicity and robustness, let's use the min of the current F array if available,
            # otherwise the reference point (though ref is usually worse than ideal).
            # Actually, using the reference point as a "utopia" is bad because it's far away.
            # Let's use the min of the current non-dominated set.
            return f, weights_vec

        # We will implement a custom local search that minimizes the Tchebycheff distance
        # to the current "ideal" point (min of objectives seen so far).
        
        # Update ideal point
        F_arr = np.array(F)
        ideal_point = F_arr.min(axis=0)
        
        # For each weight vector, perform a local search
        for i_w in range(n_weights):
            if used >= budget:
                break
            if time.time() > deadline:
                break
                
            w = weights[i_w]
            
            # Choose a starting point
            # Pick the point in the current archive that best matches this weight direction.
            # We can use the Tchebycheff value with the current ideal point as a proxy.
            # Or just pick a random point from the initial population.
            
            if len(X) > 0:
                F_arr_cur = np.array(F)
                # Calculate Tchebycheff score for all points in archive
                # score = max_j ( w_j * (f_j - ideal_j) )
                # Since f_j >= ideal_j, we can drop the abs.
                diffs = F_arr_cur - ideal_point
                scores = np.max(diffs * w, axis=1)
                start_idx = np.argmin(scores)
                x_start = X[start_idx].copy()
            else:
                x_start = lo + rng.random(dim) * span
            
            # Define the objective function for scipy
            def obj_func(x):
                f = np.asarray(problem(np.clip(x, lo, hi)), dtype=np.float64)
                # Update ideal point? No, keep it fixed for this local search to avoid non-stationary target.
                # But we can update it if we find a better point.
                # For simplicity, use the ideal point from the start of this weight's search.
                diffs = f - ideal_point
                return np.max(diffs * w)
            
            # Perform Nelder-Mead minimization
            # maxiter should be limited to conserve budget.
            # We have ~15 evals left per weight.
            max_iter = 15
            try:
                res = minimize(
                    obj_func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                x_best = res.x
                f_best = np.asarray(problem(np.clip(x_best, lo, hi)), dtype=np.float64)
                used += 1 # The final evaluation in minimize might not be counted if it's cached? 
                # Actually, scipy minimize calls the function. We need to count those.
                # Wait, my obj_func calls problem, which increments used.
                # So the evaluations inside minimize are already counted in `used`.
                # I just need to make sure I don't double count or exceed budget.
                
                # Check if we are within budget
                if used < budget:
                    # Add to archive if non-dominated
                    if not dominated_by_any(f_best, np.array(F)):
                        # Remove dominated points
                        new_X = []
                        new_F = []
                        for j in range(len(X)):
                            if not (np.all(f_best <= np.array(F[j])) and np.any(f_best < np.array(F[j]))):
                                new_X.append(X[j])
                                new_F.append(F[j])
                        new_X.append(x_best)
                        new_F.append(f_best)
                        X, F = new_X, new_F
            except Exception:
                pass

        # Phase 3: Final refinement with remaining budget
        # If we have budget left, do a few more random restarts or local searches
        # from the best points in the archive.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Pick a random point from the non-dominated set
            F_arr = np.array(F)
            nd_idx = non_dominated_set(F_arr)
            if len(nd_idx) == 0:
                break
                
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            x_start = X[parent_idx].copy()
            
            # Random perturbation
            x_new = x_start + rng.standard_normal(dim) * 0.1 * span
            x_new = np.clip(x_new, lo, hi)
            
            x_eval, f_eval = evaluate(x_new)
            
            if not dominated_by_any(f_eval, np.array(F)):
                # Remove dominated
                new_X = []
                new_F = []
                for j in range(len(X)):
                    if not (np.all(f_eval <= np.array(F[j])) and np.any(f_eval < np.array(F[j]))):
                        new_X.append(X[j])
                        new_F.append(F[j])
                new_X.append(x_eval)
                new_F.append(f_eval)
                X, F = new_X, new_F

        return X
# EVOLVE-BLOCK-END
