# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Grid-based decomposition with targeted local refinement for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 310s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        evals = 0
        X = []
        F = []

        def evaluate(x):
            nonlocal evals
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            evals += 1
            return x, f

        def is_time_left():
            return time.time() < deadline

        # Phase 1: Coarse grid exploration
        # For 4D, use a 3x3x3x3 grid = 81 points (if budget allows)
        # Adjust grid size based on budget
        if budget >= 150:
            grid_n = 3  # 3^4 = 81 points
        elif budget >= 80:
            grid_n = 2  # 2^4 = 16 points
        else:
            grid_n = 1  # just corners

        if grid_n > 1:
            grid_vals = np.linspace(lo, hi, grid_n, axis=0)
            grid_points = np.array(list(np.ndindex(*([grid_n] * dim))))
            grid_points = grid_points * (hi - lo) / (grid_n - 1) + lo

            for gp in grid_points:
                if evals >= budget or not is_time_left():
                    break
                x, f = evaluate(gp)
                X.append(x)
                F.append(f)

        # If we still have budget, add some random points for diversity
        if evals < budget:
            rng = np.random.default_rng(42)
            n_rand = min(budget - evals, max(10, budget // 5))
            for _ in range(n_rand):
                if evals >= budget or not is_time_left():
                    break
                x = lo + (hi - lo) * rng.random(dim)
                x, f = evaluate(x)
                X.append(x)
                F.append(f)

        if not X:
            # Fallback: at least one evaluation
            x = (lo + hi) / 2
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        # Phase 2: Identify promising regions and refine with Nelder-Mead
        # Use weighted-sum decomposition with multiple weight vectors
        # For 3 objectives, use a set of weight vectors

        # Generate weight vectors on the simplex
        # For 3 objectives: w1 + w2 + w3 = 1, wi >= 0
        # Use a set of evenly spaced weight vectors
        n_weights = min(12, max(3, (budget - evals) // 10))
        
        # Generate weight vectors
        weight_vectors = []
        if n_obj == 3:
            # Use a stratified set of weight vectors
            # Include extreme points and mid-points
            w_list = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [1/3, 1/3, 1/3],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
            ]
            weight_vectors = w_list[:n_weights]
        else:
            # For other n_obj, use random weights
            rng_w = np.random.default_rng(123)
            for _ in range(n_weights):
                w = rng_w.random(n_obj)
                w /= w.sum()
                weight_vectors.append(w)

        # For each weight vector, find a starting point from current population
        # and run Nelder-Mead to minimize the weighted sum
        for w in weight_vectors:
            if evals >= budget or not is_time_left():
                break

            # Find the best starting point for this weight direction
            w = np.asarray(w, dtype=float)
            scores = F @ w
            start_idx = np.argmin(scores)
            start_point = X[start_idx].copy()

            # Also try starting from a few other points
            candidates = [start_point]
            # Add a few random perturbations
            rng_c = np.random.default_rng(hash(tuple(w)) % (2**32))
            for _ in range(2):
                cand = start_point + 0.1 * (hi - lo) * rng_c.standard_normal(dim)
                cand = np.clip(cand, lo, hi)
                candidates.append(cand)

            best_x = None
            best_score = np.inf

            for cand in candidates:
                if evals >= budget or not is_time_left():
                    break

                # Run Nelder-Mead
                def objective(x):
                    nonlocal evals
                    x = np.clip(x, lo, hi)
                    f = np.asarray(problem(x), dtype=float)
                    evals += 1
                    return np.dot(w, f)

                try:
                    result = minimize(
                        objective,
                        cand,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    if result.fun < best_score:
                        best_score = result.fun
                        best_x = result.x
                except Exception:
                    pass

            if best_x is not None:
                best_x = np.clip(best_x, lo, hi)
                # Evaluate the final point to get exact objective values
                bx, bf = evaluate(best_x)
                X.append(bx)
                F.append(bf)

        # Phase 3: If budget remains, do additional local refinement
        # Pick the current Pareto-optimal points and try to improve them
        if evals < budget and is_time_left():
            # Find non-dominated points
            F_arr = np.array(F)
            X_arr = np.array(X)
            
            # Compute non-dominated set
            n_points = len(F_arr)
            non_dominated = np.ones(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            non_dominated[i] = False
                            break
            
            nd_indices = np.where(non_dominated)[0]
            
            # Try to refine each non-dominated point
            for idx in nd_indices:
                if evals >= budget or not is_time_left():
                    break
                
                start_point = X_arr[idx].copy()
                
                # Try a small local search with random perturbations
                rng_l = np.random.default_rng(idx + 999)
                best_x = start_point
                best_f = F_arr[idx].copy()
                
                for _ in range(5):
                    if evals >= budget or not is_time_left():
                        break
                    perturbation = 0.05 * (hi - lo) * rng_l.standard_normal(dim)
                    new_x = np.clip(start_point + perturbation, lo, hi)
                    nx, nf = evaluate(new_x)
                    # Check if new point is non-dominated by current best
                    if np.all(nf <= best_f) and np.any(nf < best_f):
                        best_f = nf
                        best_x = nx
                    elif np.all(best_f <= nf) and np.any(best_f < nf):
                        pass
                    else:
                        # Keep track of this point anyway
                        pass
                
                if np.all(best_f <= F_arr[idx]) and np.any(best_f < F_arr[idx]):
                    X.append(best_x)
                    F.append(best_f)

        return X
# EVOLVE-BLOCK-END
