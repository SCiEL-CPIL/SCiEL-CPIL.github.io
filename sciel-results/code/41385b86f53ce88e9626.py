"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Nelder-Mead Simplex for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            if call_count >= budget:
                return np.inf
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead Simplex Implementation
        def nelder_mead(func_eval, x0, budget_limit, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal call_count
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            f_vals = np.zeros(n + 1)
            f_vals[0] = func_eval(simplex[0])
            
            for i in range(1, n + 1):
                simplex[i] = simplex[0].copy()
                simplex[i, i - 1] += 0.5
                f_vals[i] = func_eval(simplex[i])
                
            if call_count >= budget_limit:
                return simplex[np.argmin(f_vals)], np.min(f_vals)

            for _ in range(max_iter):
                if call_count >= budget_limit or time.time() > deadline:
                    break
                    
                # Sort simplex by function value
                idx = np.argsort(f_vals)
                simplex = simplex[idx]
                f_vals = f_vals[idx]
                
                # Check convergence
                if np.ptp(f_vals) < 1e-10:
                    break
                    
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = func_eval(x_r)
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = func_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = func_eval(x_c)
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func_eval(simplex[i])
                        if call_count >= budget_limit:
                            break

            return simplex[np.argmin(f_vals)], np.min(f_vals)

        # Phase 1: Coarse Random Search to find promising basins
        # Use a portion of the budget for global exploration
        # Heuristic: Use ~20% of budget for initial random sampling, rest for local refinement
        # With 500 budget, 100 random starts is too many. Let's try 20-30 random points first.
        # Actually, for 21 peaks, we want to hit several basins.
        # Let's use a deterministic grid mixed with random jitter for the first ~50 calls.
        
        # Strategy:
        # 1. Evaluate origin.
        # 2. Generate a set of candidate starting points.
        # 3. For each candidate, if we have budget, run Nelder-Mead from there.
        # 4. Keep the best result.
        
        # To maximize AOCC, we want rapid improvement.
        # Local search converges very fast.
        
        # Initial point
        x_start = np.zeros(dim)
        evaluate(x_start)
        
        # Generate random starting points
        # We will try to launch Nelder-Mead from several points.
        # How many? Each NM run takes ~2*dim calls to init + iterations.
        # dim=5, init=6 calls.
        # If we spend 10 calls per NM run, we can do ~40 runs.
        # Let's do a mix: 
        # - 10 random points in [-4, 4]^5
        # - 10 random points in [-2, 2]^5 (closer to center, assuming peaks are spread but center is often good)
        
        np.random.seed(42) # Deterministic for reproducibility, though evaluator might not care
        
        candidates = []
        # Random points in wide box
        for _ in range(15):
            candidates.append(np.random.uniform(-4, 4, dim))
        # Random points in narrow box
        for _ in range(15):
            candidates.append(np.random.uniform(-2, 2, dim))
        
        # Sort candidates by distance to current best (optional, but let's just iterate)
        # Actually, let's interleave: evaluate a point, then run NM if it's good enough?
        # No, NM needs a good start. Let's just run NM from each candidate if budget allows.
        
        # We need to be careful with the budget.
        # Each NM call uses `evaluate` which increments `call_count`.
        
        for i, c in enumerate(candidates):
            if time.time() > deadline:
                break
            if call_count > budget - 20: # Leave some buffer
                break
                
            # Evaluate the candidate itself first
            f_cand = evaluate(c)
            
            # If this candidate is significantly better than current best, or just to explore,
            # run Nelder-Mead from here.
            # To save budget, only run NM if we have enough budget left for a reasonable run.
            # A minimal NM run takes ~10-20 calls.
            
            if call_count < budget - 15:
                # Run Nelder-Mead
                # Limit iterations to avoid wasting budget
                # Max iter depends on remaining budget
                remaining = budget - call_count
                max_iter = min(50, remaining // 2)
                
                if max_iter > 5:
                    x_opt, f_opt = nelder_mead(evaluate, c, budget, max_iter=max_iter)
                    
        # If we have budget left, do a final polish on the best point found
        if call_count < budget - 10 and best_f < np.inf:
            # Run NM again from the best found so far with tighter convergence
            remaining = budget - call_count
            max_iter = min(100, remaining // 2)
            if max_iter > 5:
                nelder_mead(evaluate, best_x, budget, max_iter=max_iter)

        return best_x, best_f

# EVOLVE-BLOCK-END
