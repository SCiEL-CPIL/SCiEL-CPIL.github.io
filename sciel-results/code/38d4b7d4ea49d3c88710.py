"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables).
    Strategy:
    1. Initial broad sampling using Sobol quasi-random sequences.
    2. Identify non-dominated points.
    3. Local refinement using L-BFGS-B on weighted sum scalarizations.
    4. Strict evaluation counting and time-guarded execution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s for our logic to be safe.
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Evaluation counter
        eval_count = 0
        max_evals = self.budget

        # Wrapper for problem to count evaluations
        def eval_problem(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return np.array(problem(x))

        X_list = []
        F_list = []

        # Phase 1: Initial Sampling
        # Use Sobol for better space filling.
        n_initial = 100
        # Use a fixed seed for reproducibility within the run, but the evaluator sets np.random.seed
        # We can use qmc.Sobol with a fixed seed or just use np.random if we want to leverage the evaluator's seed.
        # The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # If we use np.random, the results will differ by seed.
        # If we use fixed seed Sobol, results are deterministic.
        # Deterministic is usually safer for stability, but leveraging the seed might help if the problem is random.
        # RE34 is a fixed mathematical problem, so deterministic is fine.
        # However, using the evaluator's seed might be required for "fairness" or if the problem instance varies?
        # The contract says "fixed uniform random designs" for the reference, implying the problem is fixed.
        # Let's use a fixed seed for Sobol to ensure consistent quality, or use np.random to respect the seed.
        # Let's use np.random for the initial points to respect the seed mechanism, but Sobol is better.
        # scipy.stats.qmc.Sobol has a seed parameter. If we don't set it, it might use global state?
        # Let's just use np.random.uniform for simplicity and to respect the seed, or Sobol with seed=seed?
        # The evaluator sets np.random.seed(seed).
        # Let's use Sobol with a fixed seed (e.g., 42) to ensure good space filling regardless of the evaluator seed.
        # Actually, if the evaluator seed is meant to introduce variability, using a fixed seed might be "cheating" or just consistent.
        # Let's stick to fixed Sobol seed for robustness.

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            samples = sampler.random(n_initial)
        except Exception:
            samples = np.random.rand(n_initial, self.dim)

        X_init = lo + samples * span

        for x in X_init:
            if eval_count >= max_evals:
                break
            f = eval_problem(x)
            X_list.append(x)
            F_list.append(f)

        # Helper functions for dominance
        def is_dominated(f, F_set):
            # f is dominated if there exists g in F_set such that g <= f and g != f
            # Vectorized check
            diffs = F_set - f
            return np.any(np.all(diffs <= 0, axis=1) & np.any(diffs < 0, axis=1))

        def get_non_dominated_indices(F_arr):
            n = len(F_arr)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                # Check if i is dominated by any j != i
                # We can vectorize against the whole set, but exclude i
                # Simpler: loop
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Phase 2: Local Refinement
        # Generate weight vectors for scalarization
        # We want to cover the Pareto front.
        # For 3 objectives, we can use a grid of weights on the simplex.
        # Let's generate a set of diverse weights.
        
        # Vertices
        weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        # Edges (midpoints and quarters)
        for i in [0.25, 0.5, 0.75]:
            weights.append([i, 1-i, 0.0])
            weights.append([i, 0.0, 1-i])
            weights.append([0.0, i, 1-i])
        
        # Center
        weights.append([1/3, 1/3, 1/3])
        
        # Some random-ish points on the simplex for diversity
        rng = np.random.default_rng(123)
        for _ in range(10):
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)

        weights = np.array(weights)
        
        # Shuffle weights to mix order
        rng.shuffle(weights)

        # For each weight, find the best starting point from current non-dominated set
        # and refine it.
        
        F_arr = np.array(F_list) if F_list else np.array([])
        
        if len(F_arr) > 0:
            nd_indices = get_non_dominated_indices(F_arr)
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F_list))
        else:
            nd_indices = np.array([])

        for w in weights:
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break

            # Find best point for this weight in current set
            if len(nd_indices) > 0:
                scores = F_arr[nd_indices] @ w
                best_idx_in_nd = nd_indices[np.argmin(scores)]
                x_start = X_list[best_idx_in_nd]
            else:
                x_start = (lo + hi) / 2

            # Define scalarized objective
            def obj_scalar(x):
                fx = eval_problem(x)
                return float(np.dot(fx, w))

            # Try to minimize
            try:
                res = minimize(
                    obj_scalar, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                x_opt = res.x
                # The optimizer might have called eval_problem multiple times.
                # We need to record the final point.
                # Note: eval_problem inside obj_scalar increments eval_count.
                # We should record the last evaluation or the result of res.x.
                # Let's evaluate res.x explicitly to ensure it's in our list?
                # No, that's an extra eval. The last eval in L-BFGS-B is likely at x_opt or close.
                # But to be safe and consistent, let's just use the result.
                # However, we don't have the F value of res.x directly unless we call problem again.
                # Calling problem again costs 1 eval.
                # Is it worth it? Yes, for a clean record.
                
                if eval_count < max_evals:
                    f_opt = eval_problem(x_opt)
                    X_list.append(x_opt)
                    F_list.append(f_opt)
                    
                    # Update F_arr for next iteration?
                    # It's expensive to recompute non-dominated set every time.
                    # We can do it at the end or periodically.
                    # For now, let's just add to list.
                    
            except Exception:
                pass

        # Phase 3: Final Cleanup / Additional Random Sampling if budget remains
        # If we have budget left, do some random sampling to fill gaps.
        while eval_count < max_evals and time.time() < deadline:
            x_rand = lo + np.random.rand(self.dim) * span
            f_rand = eval_problem(x_rand)
            X_list.append(x_rand)
            F_list.append(f_rand)

        return X_list

# EVOLVE-BLOCK-END
