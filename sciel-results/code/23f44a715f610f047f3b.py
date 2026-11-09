"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nelder_mead(func, x0, budget, dim, deadline, best_x, best_f, seed=0, alpha=1.0, gamma=2.0, rho=0.5, sigma=1.0):
    """
    Nelder-Mead simplex method.
    Returns (best_x, best_f, calls_used).
    """
    rng = np.random.default_rng(seed)
    n = dim
    x = np.asarray(x0, dtype=float).copy()
    
    # Initialize simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        # Standard Nelder-Mead initialization: move along axis i
        # Use a small step, but scale it if the function seems to have a large scale
        # Here we use a heuristic step size relative to the domain
        step = 0.25 
        xi[i] += step
        simplex.append(xi)
        
    # Evaluate initial simplex
    fvals = np.empty(n + 1)
    calls = 0
    for i in range(n + 1):
        if time.time() > deadline:
            break
        fvals[i] = func(simplex[i])
        calls += 1
        if fvals[i] < best_f:
            best_f, best_x = fvals[i], simplex[i].copy()

    # Order simplex by fvals
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = fvals[order]

    while calls < budget:
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.std(fvals) < 1e-10:
            break
            
        # Centroid of all but the worst point
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_f = fvals[-1]

        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = func(xr)
        calls += 1
        if fr < best_f:
            best_f, best_x = fr, xr.copy()

        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - worst)
            fe = func(xe)
            calls += 1
            if fe < best_f:
                best_f, best_x = fe, xe.copy()
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < worst_f:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f, best_x = fc, xc.copy()
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f, best_x = fvals[i], simplex[i].copy()
            else:
                # Inside contraction
                xc = centroid - 0.5 * (centroid - worst)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f, best_x = fc, xc.copy()
                if fc <= worst_f:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f, best_x = fvals[i], simplex[i].copy()
                            
        # Re-order simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]
        
        # Prevent infinite loops on non-finite values
        if not np.all(np.isfinite(fvals)):
            break

    return best_x, best_f, calls


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Strategy: Multi-start Nelder-Mead
        # We divide the budget into K restarts.
        # K=10 seems reasonable for dim=5 and budget=500.
        # Each restart gets budget/K calls.
        # We also include a "global" random scan at the very beginning to pick good starts.
        
        # Phase 1: Quick random scan to identify a few good starting points
        # Use 10% of budget for this? Or just a few points.
        # Let's use 20 points for initial scan.
        n_scan = 20
        if budget > 50:
             n_scan = min(20, budget // 10)
        else:
             n_scan = 5
             
        rng = np.random.default_rng(42)
        
        # Candidate starts: Origin, +/- corners, and random
        starts = [np.zeros(dim)]
        # Add some structured points
        starts.append(np.array([1.0] * dim))
        starts.append(np.array([-1.0] * dim))
        starts.append(np.array([2.0] * dim))
        starts.append(np.array([-2.0] * dim))
        
        while len(starts) < n_scan:
            starts.append(rng.uniform(-4, 4, dim))
            
        # Evaluate starts
        start_fvals = []
        for s in starts:
            if time.time() > deadline:
                break
            fs = func(np.clip(s, -5, 5))
            calls += 1
            start_fvals.append(fs)
            if fs < best_f:
                best_f, best_x = fs, np.clip(s, -5, 5).copy()
                
        # Sort starts by function value
        sorted_indices = np.argsort(start_fvals)
        
        # Phase 2: Run Nelder-Mead from the best K starts
        # Remaining budget
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f
            
        # Number of local searches
        # If we have 500 calls, and used 20 for scan, we have 480.
        # Let's do 8 local searches of ~60 calls each.
        # Or adaptively: if we find a very good point, spend more there.
        # Simple approach: Top 4 starts get larger budget, next 4 get smaller.
        
        num_local = 8
        if remaining < num_local * 10:
            num_local = max(1, remaining // 10)
            
        # Allocate budget to local searches
        # Weighted by how good the start was?
        # Or just equal split?
        # Let's do equal split for simplicity and robustness.
        budget_per_local = remaining // num_local
        
        # We will run Nelder-Mead sequentially.
        # To maximize AOCC, we want to improve the best value early.
        # So we run the best start first, then the next best, etc.
        
        for i in range(num_local):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
                
            # Pick a start. Use the i-th best start from the scan.
            # If i < len(sorted_indices), use that. Otherwise, random.
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                x0 = np.clip(starts[idx], -5, 5)
            else:
                x0 = rng.uniform(-4, 4, dim)
                
            # Run Nelder-Mead
            # Pass current best to update global best
            best_x, best_f, c_used = _nelder_mead(
                func, x0, budget_per_local, dim, deadline, 
                best_x, best_f, seed=100 + i
            )
            calls += c_used
            
        # If we have leftover budget, do a final polish on the best point found
        if calls < budget and time.time() < deadline:
             remaining = budget - calls
             if remaining > 5:
                 best_x, best_f, c_used = _nelder_mead(
                     func, best_x, remaining, dim, deadline,
                     best_x, best_f, seed=999
                 )
                 calls += c_used

        return best_x, best_f
# EVOLVE-BLOCK-END
