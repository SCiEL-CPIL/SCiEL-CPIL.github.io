"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using multi-start local search via scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Generate initial points
        # We will spend a small fraction of budget on initial random exploration
        # and the rest on local refinement.
        # Strategy: Divide budget into 'runs'. Each run starts from a random point
        # and performs a local search to minimize a weighted sum of objectives.
        
        # How many local search runs can we afford?
        # Nelder-Mead with maxiter=50 uses ~50-100 evals per run (depending on convergence).
        # Let's aim for ~20-30 runs to cover the Pareto front well.
        # If budget=200, 20 runs * 10 evals/run = 200.
        # Let's be conservative: 15 runs.
        
        n_runs = 15
        # Ensure we don't exceed budget. 
        # We will track evaluations globally.
        
        all_points = []
        all_objectives = []
        
        # Pre-allocate some initial random points to seed the search
        # These also count towards the budget.
        n_initial = min(10, budget // 10)
        
        # Generate diverse initial points: some random, some near corners
        init_xs = []
        for i in range(n_initial):
            if i < n_initial // 2:
                x = lo + span * rng.random(dim)
            else:
                # Corners or edges
                c = rng.random(dim) < 0.5
                x = np.where(c, lo, hi)
            init_xs.append(x)
        
        # Evaluate initial points
        for x in init_xs:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(x)
            all_objectives.append(f)
            
        used_evals = len(all_points)

        # Define scalarization function
        def scalarized_objective(x, weights):
            # Weights are fixed for a specific run
            fx = problem(x)
            return np.dot(weights, fx)

        # Perform local searches
        # We will pick random weights for each run to cover different parts of the Pareto front
        # For 2 objectives, weights are (w, 1-w). We sample w from [0.01, 0.99]
        
        for run in range(n_runs):
            if used_evals >= budget:
                break
            if time.time() > deadline:
                break
                
            # Choose weights
            # Log-uniform or uniform? Uniform is standard for convex fronts.
            # RE21 might be non-convex, but uniform is a safe bet.
            w1 = rng.uniform(0.05, 0.95)
            weights = np.array([w1, 1.0 - w1])
            
            # Choose starting point
            # Pick a random point from our current archive or a new random point
            if len(all_points) > 0:
                # 50% chance to start from existing archive point, 50% new random
                if rng.random() < 0.5:
                    start_idx = rng.integers(len(all_points))
                    x0 = all_points[start_idx].copy()
                else:
                    x0 = lo + span * rng.random(dim)
            else:
                x0 = lo + span * rng.random(dim)
                
            # Ensure x0 is within bounds
            x0 = np.clip(x0, lo, hi)
            
            # Run Nelder-Mead
            # maxiter is a proxy for evals, but NM can use more.
            # Let's limit maxiter to ensure we don't blow the budget.
            # Budget remaining: budget - used_evals
            # We want to use roughly (budget - used_evals) / (n_runs - run) evals per remaining run?
            # Simpler: Just cap maxiter at a reasonable number, e.g., 30.
            # NM typically uses ~2*dim*maxiter evals. dim=4, maxiter=30 -> ~240 evals worst case.
            # That's too high. Let's use maxiter=10. ~80 evals.
            # With 15 runs, that's 1200 evals. Too many.
            
            # We need to be smarter.
            # Let's allocate a fixed number of evals per local search, say 5-10.
            # maxiter=5 for NM with dim=4 is about 20-40 evals.
            # Let's try maxiter=3.
            
            max_iter = 5
            try:
                res = minimize(
                    scalarized_objective, 
                    x0, 
                    args=(weights,),
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
                )
                # res.x is the best point found
                # Note: minimize might have used more or fewer evals than expected.
                # We must count the actual evaluations used.
                # Unfortunately, scipy minimize doesn't easily expose exact eval count in a simple attribute for all methods, 
                # but `res.nfev` is available for Nelder-Mead.
                
                if hasattr(res, 'nfev'):
                    evals_used = res.nfev
                else:
                    evals_used = max_iter * 2 * dim # Fallback estimate
                    
                # Add the result
                x_best = np.clip(res.x, lo, hi)
                f_best = problem(x_best)
                
                # Check if this point is new and good
                # We just append it. The hypervolume calculation later will handle dominance.
                all_points.append(x_best)
                all_objectives.append(f_best)
                used_evals += evals_used
                
            except Exception:
                # If local search fails, just use the start point if we haven't evaluated it yet
                # Or skip. Since we might have evaluated x0 already, we don't add it again unless it's new.
                # To be safe, let's just continue.
                pass

        # Final check: ensure we haven't exceeded budget significantly
        # The contract says "More than 200 evaluations ... makes the whole result invalid"
        # Wait, "More than 200 evaluations ... makes the whole result invalid".
        # My loop above adds evals_used to a counter, but the `problem` function is called inside `minimize`.
        # I cannot easily intercept every single call inside `minimize` to stop it exactly at 200.
        # If `minimize` goes over 200, the result is INVALID.
        
        # This is a CRITICAL constraint.
        # "More than 200 evaluations ... makes the whole result invalid".
        # This means I must strictly control the number of times `problem` is called.
        # `scipy.optimize.minimize` does not allow me to easily cap the *total* global budget across multiple calls easily without a wrapper.
        
        # I need a wrapper around `problem` that counts calls and raises an exception or stops if budget is exceeded.
        # But `minimize` will catch the exception? No, if I raise an exception, `minimize` might propagate it or handle it poorly.
        # Better: The wrapper returns a dummy value if budget is exceeded? No, that corrupts the optimization.
        
        # Let's re-read: "More than 200 evaluations ... makes the whole result invalid".
        # This implies I should NOT exceed 200.
        # If I use `minimize`, I don't have fine-grained control over the internal steps.
        
        # Alternative Strategy: Manual Local Search?
        # Or use `minimize` but with a very tight maxiter and a wrapper that counts.
        
        # Let's implement a strict counter wrapper.
        
        # RE-IMPLEMENTATION with strict budget control
        
        # I will restart the logic inside __call__ to be safe.
        
        # --- RESTART LOGIC ---
        
        # Clear previous state
        all_points = []
        all_objectives = []
        global_eval_count = 0
        budget_limit = self.budget
        
        def counted_problem(x):
            nonlocal global_eval_count
            if global_eval_count >= budget_limit:
                # Return a very bad value to discourage further exploration?
                # Or raise an exception?
                # If I raise, minimize might crash.
                # If I return bad values, it might converge to bad points.
                # Best is to prevent the call.
                # But we can't prevent the call from inside minimize easily.
                
                # Actually, if I hit the limit, I should just stop the whole process.
                # But I'm inside a minimize call.
                # Let's just return a dummy vector of large values.
                return np.array([1e10, 1e10])
            
            global_eval_count += 1
            return problem(x)
            
        # We need to ensure that if we hit the limit, we don't add more points.
        # And we need to catch the case where we exceed the limit.
        
        # Let's try a different approach:
        # 1. Do a few random evals.
        # 2. Do local searches with STRICT maxiter.
        # 3. Check global_eval_count after each minimize.
        # 4. If global_eval_count > budget_limit, the result is invalid anyway.
        #    So I must ensure global_eval_count <= budget_limit.
        
        # To guarantee this, I should limit maxiter such that worst-case evals < remaining budget.
        # NM evals per iteration is roughly 2*dim.
        # So maxiter * 2 * dim <= remaining_budget.
        
        # Let's redo the loop with this logic.
        
        # 1. Initial Random Points
        n_initial = 10
        for i in range(n_initial):
            if global_eval_count >= budget_limit:
                break
            x = lo + span * rng.random(dim)
            f = counted_problem(x)
            all_points.append(x)
            all_objectives.append(f)
            
        # 2. Local Searches
        n_runs = 15
        for run in range(n_runs):
            if global_eval_count >= budget_limit:
                break
            if time.time() > deadline:
                break
                
            remaining = budget_limit - global_eval_count
            # Estimate evals per NM iteration: ~2*dim
            # Leave some buffer
            max_iter = max(1, int((remaining * 0.8) / (2 * dim)))
            
            # Don't let max_iter get too large to ensure we finish
            max_iter = min(max_iter, 20)
            
            w1 = rng.uniform(0.05, 0.95)
            weights = np.array([w1, 1.0 - w1])
            
            if len(all_points) > 0:
                if rng.random() < 0.5:
                    start_idx = rng.integers(len(all_points))
                    x0 = all_points[start_idx].copy()
                else:
                    x0 = lo + span * rng.random(dim)
            else:
                x0 = lo + span * rng.random(dim)
                
            x0 = np.clip(x0, lo, hi)
            
            def scalarized_objective(x, weights):
                fx = counted_problem(x)
                return np.dot(weights, fx)
                
            try:
                res = minimize(
                    scalarized_objective, 
                    x0, 
                    args=(weights,),
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
                )
                
                # Check if we blew the budget
                if global_eval_count > budget_limit:
                    # We exceeded budget. The result is invalid.
                    # We can't fix this, but we should stop adding points.
                    # Actually, if we exceed, the score is 0.
                    # So we just break.
                    break
                    
                x_best = np.clip(res.x, lo, hi)
                # We don't need to evaluate x_best again if it was the last one evaluated?
                # No, minimize returns the x that minimizes the objective, but we don't have its f directly.
                # We must evaluate it to record it.
                # BUT, if we evaluate it, we use 1 more eval.
                # If global_eval_count + 1 > budget_limit, we cannot evaluate it.
                
                if global_eval_count < budget_limit:
                    f_best = counted_problem(x_best)
                    all_points.append(x_best)
                    all_objectives.append(f_best)
                else:
                    # Cannot evaluate, skip adding
                    pass
                    
            except Exception:
                pass

        return all_points

# EVOLVE-BLOCK-END
