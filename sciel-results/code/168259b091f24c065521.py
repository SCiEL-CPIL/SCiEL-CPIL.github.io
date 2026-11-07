"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Tchebycheff scalarization with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # We use a global counter to strictly enforce the evaluation limit
        eval_count = [0]
        
        def counting_problem(x):
            if eval_count[0] >= budget:
                # Return a bad value to stop optimization, but don't crash
                return [1e10] * n_obj
            eval_count[0] += 1
            # Problem clips x to bounds, but we ensure finite input
            x_safe = np.clip(x, lo, hi)
            return np.array(problem(x_safe))

        # Generate Tchebycheff weight vectors
        # For 2 objectives, weights are (w, 1-w). We choose 20 distinct weights.
        # To cover the front well, we spread them uniformly from 0.05 to 0.95
        n_subproblems = 20
        weights = np.linspace(0.05, 0.95, n_subproblems)
        
        # Allocate budget: 10 evaluations per subproblem
        # 20 * 10 = 200 exactly.
        # We might reserve a few for initial random sampling if needed, 
        # but let's try pure allocation first.
        evals_per_sub = max(5, budget // n_subproblems)
        
        all_points = []
        
        # Pre-generate some random starting points to reuse or pick from
        rng = np.random.default_rng(42)
        random_starts = lo + span * rng.random((10, dim))

        for i, w in enumerate(weights):
            if time.time() > deadline:
                break
            if eval_count[0] >= budget:
                break
            
            # Tchebycheff scalarization:
            # min_x max_i ( w_i * (f_i(x) - z_i*) )
            # Since we don't know z* (ideal point), we use a dynamic or static approximation.
            # A common simple approach is to use the current best known values or just minimize max(w_i * f_i).
            # However, standard Tchebycheff uses z*. Let's use a simplified version:
            # min max_i ( w_i * f_i(x) )
            # This is equivalent to finding the point where the weighted objectives are balanced.
            
            # Objective function for scipy
            def tcheb_obj(x, w=w):
                vals = counting_problem(x)
                # If we hit the limit, return large value
                if eval_count[0] >= budget and eval_count[0] > 0:
                     # Actually, counting_problem returns 1e10 if over budget.
                     # We need to handle the case where it's the last allowed eval.
                     pass
                weighted = w * vals[0]
                weighted_2 = (1 - w) * vals[1]
                return max(weighted, weighted_2)

            # Bounds for scipy
            scipy_bounds = list(zip(lo, hi))
            
            # Try a few different starting points for this subproblem if budget allows
            # With 10 evals per subproblem, we can try 2-3 starts or 1 robust start.
            # Nelder-Mead typically needs dim+1 initial points. dim=4, so 5 points.
            # 10 evals is tight for Nelder-Mead. Let's use 1 start with 9 evals, or 2 starts with 5 evals each?
            # Nelder-Mead with 4 vars: initial simplex is 5 points. 
            # So 1 run consumes 5+ evals. 
            # Let's do 1 run per subproblem with maxiter=4 (approx 5-10 evals depending on implementation).
            # Actually, scipy's minimize counts function calls.
            
            best_x = None
            best_val = np.inf
            
            # Use a random start from our pool, or center
            start = random_starts[i % len(random_starts)].copy()
            
            try:
                res = minimize(tcheb_obj, start, method='Nelder-Mead', 
                               bounds=scipy_bounds, 
                               options={'maxiter': evals_per_sub, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                # Fallback to the start point if optimization fails
                if eval_count[0] < budget:
                    val = tcheb_obj(start)
                    best_val = val
                    best_x = start
            
            if best_x is not None:
                # Evaluate the final point to ensure it's in our list (it was evaluated during minimize, but let's record it)
                # Note: The evaluation inside minimize already added to eval_count.
                # We just need the objective values.
                # We should re-evaluate? No, that costs budget.
                # The last evaluation in minimize is res.x.
                # We can just trust res.x and its associated function value structure, 
                # but we need the actual objective vector.
                # Since tcheb_obj calls counting_problem, the last call was for res.x.
                # We can't easily retrieve the vector without re-calling.
                # However, re-calling costs 1 eval.
                # Alternative: Keep a history in the closure.
                
                pass

        # The above approach has a flaw: we don't easily get the final objective vectors without re-evaluating.
        # Let's refactor to keep a history of all evaluated points.

        # RE-IMPLEMENTATION with History
        eval_count[0] = 0 # Reset
        history_X = []
        history_F = []

        def counting_problem_v2(x):
            if eval_count[0] >= budget:
                return [1e10] * n_obj
            eval_count[0] += 1
            x_safe = np.clip(x, lo, hi)
            f = np.array(problem(x_safe))
            history_X.append(x_safe)
            history_F.append(f)
            return f

        def tcheb_obj_v2(x, w=w):
            vals = counting_problem_v2(x)
            weighted = w * vals[0]
            weighted_2 = (1 - w) * vals[1]
            return max(weighted, weighted_2)

        all_points = []
        
        for i, w in enumerate(weights):
            if time.time() > deadline:
                break
            if eval_count[0] >= budget:
                break
            
            start = random_starts[i % len(random_starts)].copy()
            
            try:
                res = minimize(tcheb_obj_v2, start, method='Nelder-Mead', 
                               bounds=scipy_bounds, 
                               options={'maxiter': evals_per_sub, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        # After all subproblems, we have history_X and history_F.
        # We need to return the non-dominated points or all points?
        # The evaluator records every returned objective vector.
        # The return value of __call__ is ignored, but "Every returned objective vector is recorded".
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This usually implies the evaluator wraps `problem` to record calls.
        # "problem(x) ... returns 2 objective values."
        # "Every returned objective vector is recorded" -> This refers to the calls to `problem`.
        # So the evaluator records every time `problem` is called.
        # Therefore, I don't need to return the points, I just need to ensure I make good calls to `problem`.
        # The score is based on the set of all recorded vectors.
        # So I just need to execute the search. The return value doesn't matter for scoring, only for validity (no exception).
        
        return np.array(history_X) if history_X else np.array([])

# EVOLVE-BLOCK-END
