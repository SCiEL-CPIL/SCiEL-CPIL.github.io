"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with (1+1)-ES refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit, use 310s to be safe
        deadline = start_time + 310.0

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)

        # Thread-safe-ish wrapper for function evaluation
        def safe_func(x):
            nonlocal calls, best_f, best_x
            # Check time
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            # Check budget
            if calls >= total_budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            
            f = func(x_clipped)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
                
            return f

        # Phase 1: Multi-start Nelder-Mead
        # We use Nelder-Mead because it is robust for Rosenbrock's valley shape
        # and doesn't require gradient estimation which can be noisy or expensive.
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # 1. Origin (often good for shifted Rosenbrock)
        # 2. Random points in the box
        # 3. Points near the boundaries (less likely but possible)
        starts = [np.zeros(n)]
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, size=n))
        
        # Budget allocation: 70% for global search (Nelder-Mead), 30% for local refinement
        nm_budget = int(total_budget * 0.7)
        es_budget = total_budget - nm_budget
        
        # Distribute NM budget among starts
        # We give more budget to the first few starts as they are more likely to succeed
        # or we just split evenly. Let's split evenly but stop early if we find a very good solution.
        budget_per_start = nm_budget // len(starts)
        
        for i, x0 in enumerate(starts):
            if calls >= nm_budget:
                break
            if time.time() > deadline:
                break
            
            # Run Nelder-Mead from x0
            # We implement a simple Nelder-Mead loop to have full control over the budget
            # SciPy's minimize might not respect the strict call count as tightly in all edge cases
            # or might use internal caching that confuses the counter if not careful.
            # However, using SciPy is faster to write and robust. Let's use a custom tight loop
            # or SciPy with a callback.
            
            # Let's use a custom Nelder-Mead implementation for strict budget control
            # and to avoid SciPy overhead if possible, though SciPy is fine.
            # Actually, a custom NM is easy for n=5.
            
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for j in range(1, n + 1):
                simplex[j] = x0.copy()
                # Initial step size 0.5 is reasonable for [-5,5]
                simplex[j, j - 1] = x0[j - 1] + 0.5
            
            fvals = np.zeros(n + 1)
            try:
                for j in range(n + 1):
                    fvals[j] = safe_func(simplex[j])
            except (StopIteration, TimeoutError):
                break
            
            # Standard NM parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            nm_calls_used = n + 1
            max_nm_iter = budget_per_start - nm_calls_used
            
            for _ in range(max_nm_iter):
                if calls >= nm_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Convergence check
                if np.abs(fvals[-1] - fvals[0]) < 1e-10:
                    break
                
                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_f = fvals[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst_point)
                xr = np.clip(xr, -5, 5)
                fr = safe_func(xr)
                nm_calls_used += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = safe_func(xe)
                    nm_calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst_point - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_func(xc)
                    nm_calls_used += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, n + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                        try:
                            fvals[j] = safe_func(simplex[j])
                            nm_calls_used += 1
                        except (StopIteration, TimeoutError):
                            break
                        if nm_calls_used > budget_per_start:
                            break
                
                # Update global best if any new point was better
                # safe_func already updates best_x/best_f

        # Phase 2: (1+1)-ES Refinement
        # Use the remaining budget to polish the best point found.
        # This is crucial for Rosenbrock to get deep into the valley.
        
        if best_x is not None and calls < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with small sigma for refinement
            rng_es = np.random.default_rng(99)
            
            while calls < total_budget:
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng_es.standard_normal(n)
                y = np.clip(y, -5, 5)
                
                try:
                    fy = safe_func(y)
                except (StopIteration, TimeoutError):
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1  # Slight increase on success
                else:
                    sigma *= 0.8  # Decrease on failure
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
