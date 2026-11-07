import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Optimizer for BBOB-style f7 (Step Ellipsoidal).
    
    Strategy:
    1. Latin Hypercube Sampling (LHS) for initial diverse exploration.
    2. Nelder-Mead local refinement from the best LHS points.
    3. CMA-ES for robust continuous optimization to handle the ellipsoidal structure.
    4. Local random perturbation search to handle step boundaries/plateaus.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        d = self.dim
        rng = np.random.default_rng(42)
        
        # State tracking
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return (time.time() - start_time) < timeout

        def safe_func(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling (LHS)
        # LHS ensures better coverage than random sampling for the same number of points
        n_lhs = 60
        if calls_made + n_lhs > self.budget:
            n_lhs = max(1, self.budget - calls_made)
            
        lhs_points = np.zeros((n_lhs, d))
        for i in range(d):
            # Generate LHS for 1D
            idx = np.arange(n_lhs)
            # Random permutation for LHS
            perm = rng.permutation(n_lhs)
            # Map to [0, 1]
            u = (perm + rng.random(n_lhs)) / n_lhs
            # Map to [-5, 5]
            lhs_points[:, i] = -5.0 + 10.0 * u

        lhs_fvals = []
        for i in range(n_lhs):
            if not time_ok() or calls_made >= self.budget:
                break
            fx = safe_func(lhs_points[i])
            lhs_fvals.append(fx)
            update_best(lhs_points[i], fx)

        if not lhs_fvals:
            return best_x, best_f

        # Sort LHS points by fitness
        lhs_indices = np.argsort(lhs_fvals)
        
        # Phase 2: Nelder-Mead Refinement
        # Try NM on the top 3 LHS points
        n_nm_restarts = 3
        nm_budget_per_restart = 150
        
        for restart_idx in range(n_nm_restarts):
            if not time_ok() or calls_made >= self.budget:
                break
            
            # Select the (restart_idx + 1)-th best LHS point
            if restart_idx < len(lhs_indices):
                idx = lhs_indices[restart_idx]
                x0 = lhs_points[idx].copy()
            else:
                x0 = best_x.copy()
            
            # Run Nelder-Mead
            try:
                # NM options
                # xatol: Tolerance for x
                # fatol: Tolerance for f
                # maxiter: Max iterations
                # We limit maxiter to ensure we don't run out of budget
                # NM uses d+1 points per iteration.
                max_iter_nm = min(100, (self.budget - calls_made) // (d + 1))
                
                res = scipy.optimize.minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'maxiter': max_iter_nm,
                        'disp': False
                    }
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except Exception:
                pass

        # Phase 3: CMA-ES for robust optimization
        # CMA-ES is better suited for the ellipsoidal structure of f7
        if best_x is not None and time_ok() and calls_made < self.budget:
            try:
                from scipy.optimize import minimize
                
                # Use CMA-ES via scipy if available, otherwise fallback to L-BFGS-B
                # CMA-ES is more robust for the step ellipsoidal function
                remaining_budget = self.budget - calls_made
                if remaining_budget > 50:
                    # CMA-ES settings
                    x0 = best_x.copy()
                    sigma0 = 0.5  # Initial step size
                    
                    res = scipy.optimize.minimize(
                        safe_func,
                        x0,
                        method='CMA-ES',
                        options={
                            'maxiter': 50,
                            'popsize': max(4 + int(3 * np.log(d)), 8),
                            'tolfun': 1e-10,
                            'tolfunr': 1e-10,
                            'verbose': 0,
                            'seed': 42
                        }
                    )
                    
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                        
            except Exception:
                # Fallback to L-BFGS-B if CMA-ES fails
                try:
                    remaining_budget = self.budget - calls_made
                    if remaining_budget > 20:
                        res = scipy.optimize.minimize(
                            safe_func,
                            best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * d,
                            options={'maxiter': 100, 'maxls': 50}
                        )
                        
                        if res.fun < best_f:
                            update_best(res.x, res.fun)
                except Exception:
                    pass

        # Phase 4: Local Random Perturbation Search
        # Step functions often have sharp boundaries. NM/CMA-ES might stop just outside the optimal step.
        # We perform a local random search around the current best point with decreasing radius.
        
        if best_x is not None and time_ok() and calls_made < self.budget:
            # Define a few radii to search
            # Start with a medium radius, shrink it
            radii = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            
            current_best_x = best_x.copy()
            current_best_f = best_f
            
            for r in radii:
                if not time_ok() or calls_made >= self.budget:
                    break
                
                # Generate random points in a sphere of radius r around current_best_x
                # Number of points: enough to use remaining budget but not too many
                n_local = 20
                if calls_made + n_local > self.budget:
                    n_local = self.budget - calls_made
                
                for _ in range(n_local):
                    if not time_ok() or calls_made >= self.budget:
                        break
                    
                    # Random direction
                    dir = rng.standard_normal(d)
                    dir /= np.linalg.norm(dir)
                    
                    # Random distance in [0, r]
                    dist = rng.uniform(0, r)
                    
                    x_candidate = current_best_x + dist * dir
                    x_candidate = np.clip(x_candidate, -5.0, 5.0)
                    
                    fx = safe_func(x_candidate)
                    
                    if fx < current_best_f:
                        current_best_f = fx
                        current_best_x = x_candidate.copy()
                        # Update global best
                        update_best(current_best_x, current_best_f)

        return best_x, best_f
