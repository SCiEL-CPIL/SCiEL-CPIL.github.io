# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 (5 variables, 3 objectives, bounds [1,3]).
    Strategy:
      1. Dense random initialization to quickly build a diverse archive.
      2. Adaptive scalarization (Tchebycheff) with local search using Nelder-Mead.
      3. Deadline guard to ensure termination before 330s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Archive: list of (x, f) pairs that are non-dominated
        archive_x = []
        archive_f = []
        evaluations_used = 0

        def is_dominated(f_new, archive_f_list):
            # Check if f_new is dominated by any point in archive_f_list
            # f_new is dominated if there exists f_a such that f_a <= f_new and f_a < f_new
            for f_a in archive_f_list:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            nonlocal evaluations_used
            # Check if new point dominates any existing points
            # Remove dominated points from archive
            new_archive_x = []
            new_archive_f = []
            for x_a, f_a in zip(archive_x, archive_f):
                # If f_new dominates f_a, skip f_a
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    continue
                # If f_a dominates f_new, skip adding f_new
                # But we check that before calling update_archive usually
                new_archive_x.append(x_a)
                new_archive_f.append(f_a)
            
            # Check if f_new is dominated by any remaining
            for f_a in new_archive_f:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return False
            
            new_archive_x.append(x_new)
            new_archive_f.append(f_new)
            
            # Replace lists
            archive_x.clear()
            archive_f.clear()
            archive_x.extend(new_archive_x)
            archive_f.extend(new_archive_f)
            return True

        # Helper to evaluate and track budget
        def evaluate(x):
            nonlocal evaluations_used
            if evaluations_used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evaluations_used += 1
            return x, f

        # Phase 1: Random Initialization
        # Use a significant portion of budget for exploration
        n_init = min(budget // 2, 100)
        # Ensure we don't exceed budget
        n_init = min(n_init, budget)
        
        # Generate random points
        # Use np.random.seed is set by evaluator, but we can use default_rng for reproducibility within run?
        # The contract says "np.random.seed(seed)" is called. So global np.random is seeded.
        # We should probably use np.random directly to respect the seed, or default_rng with a derived seed?
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # Using np.random.random will be deterministic based on that seed.
        
        for _ in range(n_init):
            if evaluations_used >= budget or time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            update_archive(x, f)

        if evaluations_used >= budget or time.time() > deadline:
            return archive_x

        # Phase 2: Local Search with Adaptive Scalarization
        # We will perform local searches from points in the archive.
        # We use Tchebycheff scalarization: min max_i (w_i * (f_i(x) - z_i^min))
        # z_i^min is the current best value for objective i.
        
        # We need to be careful with the number of evaluations used by local search.
        # Nelder-Mead uses many evaluations. We need to limit them.
        
        # Let's define a local search function that respects budget and deadline.
        def local_search(x0, max_evals=20):
            nonlocal evaluations_used
            if evaluations_used >= budget:
                return
            if time.time() > deadline:
                return
            
            # Get current z_min
            if not archive_f:
                return
            z_min = np.min(archive_f, axis=0)
            
            # Pick random weights
            w = np.random.dirichlet(np.ones(n_obj))
            
            # Objective function for scalarization
            def scalar_obj(x):
                nonlocal evaluations_used
                if evaluations_used >= budget or time.time() > deadline:
                    return 1e10
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=np.float64)
                evaluations_used += 1
                # Tchebycheff
                val = np.max(w * (f - z_min))
                return val
            
            # Initial point
            x_curr = x0.copy()
            
            # Simple Nelder-Mead implementation to avoid SciPy dependency issues if any, 
            # or just use scipy if available. Contract says SciPy is available.
            # But to be safe and fast, let's implement a simple hill climbing or use scipy.optimize.minimize.
            # Using scipy.optimize.minimize with method='Nelder-Mead'
            try:
                from scipy.optimize import minimize
                # We need to wrap the objective to check budget/deadline
                # SciPy might call the function many times. We have to stop it.
                # This is tricky because minimize doesn't have a "stop on budget" callback easily.
                # We can use a wrapper that raises an exception if budget exceeded, and catch it.
                
                class BudgetExceeded(Exception):
                    pass
                
                def obj_wrapper(x):
                    if evaluations_used >= budget or time.time() > deadline:
                        raise BudgetExceeded()
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x), dtype=np.float64)
                    evaluations_used += 1
                    val = np.max(w * (f - z_min))
                    return val
                
                # Initial simplex
                # Default simplex in scipy is based on x0 and 0.05 * (abs(x0) + 1)
                # We can specify initial simplex to be within bounds
                # Let's just use default and rely on clipping in obj_wrapper? 
                # No, Nelder-Mead moves vertices. If they go out of bounds, we clip in obj_wrapper.
                # But the simplex vertices themselves might go out of bounds.
                # It's safer to provide an initial simplex within bounds or handle it.
                # For simplicity, we'll just let it run and clip in the function.
                # However, if the simplex expands too much, it might waste evaluations.
                
                # Limit maxiter
                options = {'maxiter': 50, 'maxfev': max_evals}
                
                res = minimize(obj_wrapper, x0, method='Nelder-Mead', options=options)
                if res.success or res.fun < 1e9:
                    x_best = res.x
                    # Evaluate x_best to get its f for archive update?
                    # It was already evaluated during minimize. But we don't have the f vector stored.
                    # We must re-evaluate or store it. Re-evaluating costs 1 eval.
                    # Let's re-evaluate to be sure we have the f vector.
                    if evaluations_used < budget and time.time() < deadline:
                        x_best = np.clip(x_best, lo, hi)
                        f_best = np.array(problem(x_best), dtype=np.float64)
                        evaluations_used += 1
                        update_archive(x_best, f_best)
                        
            except BudgetExceeded:
                pass
            except Exception:
                pass

        # Iterate local searches from archive points
        # We will cycle through the archive points.
        # To ensure we don't get stuck, we can randomize the order or pick random points.
        
        # Sort archive by some metric? Or just iterate.
        # Let's iterate through the archive multiple times if budget allows.
        
        # First, let's try local search from the best points (lowest sum of objectives?)
        # Or just all points.
        
        # To avoid excessive time, we limit the number of local search attempts.
        # Each local search uses up to max_evals.
        # Let's say we allow up to 20 local search attempts.
        
        max_local_searches = 20
        search_count = 0
        
        # Create a list of indices to search
        # We can repeat the archive list to allow re-searching if needed, but usually one pass is enough.
        # Let's shuffle the archive indices for diversity.
        indices = list(range(len(archive_x)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if search_count >= max_local_searches:
                break
            if evaluations_used >= budget or time.time() > deadline:
                break
            
            # Pick a point from archive
            x_start = archive_x[idx].copy()
            local_search(x_start, max_evals=15)
            search_count += 1

        # If we still have budget, do more random restarts with local search
        while evaluations_used < budget and time.time() < deadline:
            x_rand = np.random.uniform(lo, hi)
            res = evaluate(x_rand)
            if res is None:
                break
            x, f = res
            if update_archive(x, f):
                # If new point added, maybe do local search from it?
                # But we might run out of budget.
                if evaluations_used < budget - 15 and time.time() < deadline - 5:
                    local_search(x, max_evals=15)

        return archive_x
# EVOLVE-BLOCK-END
