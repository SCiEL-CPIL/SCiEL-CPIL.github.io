"""
Deterministic multi-start LHS + accelerated BFGS local search for BBOB f19.
Replaces stochastic CMA-ES with a systematic global-to-local pipeline.
"""
import numpy as np
import time


class Optimizer:
    """Deterministic LHS multi-start with BFGS polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return np.inf
            calls_made += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def budget_ok():
            return calls_made < total_budget

        # --- Phase 1: Stratified LHS global sampling ---
        # Generate stratified Latin hypercube points for guaranteed coverage
        # Number of LHS points: use ~30-40% of budget for exploration
        n_lhs = min(20, max(10, total_budget // 15))
        
        # Create stratified LHS points
        rng = np.random.default_rng(12345)
        lhs_points = []
        
        # Standard LHS: for each dimension, divide [0,1] into n_lhs strata
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                # Stratum i in dimension d
                lo = i / n_lhs
                hi = (i + 1) / n_lhs
                point[d] = rng.uniform(lo, hi)
            # Map from [0,1]^d to [-5,5]^d
            point = point * 10 - 5
            lhs_points.append(point)
        
        # Add some structured points
        structured = [
            np.zeros(dim),
            np.full(dim, -3.0),
            np.full(dim, 3.0),
            np.full(dim, -4.0),
            np.full(dim, 4.0),
        ]
        
        # Evaluate all LHS + structured points
        all_initial = lhs_points + structured
        for p in all_initial:
            if not budget_ok() or not time_ok():
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # --- Phase 2: BFGS local search from top basins ---
        # Sort initial points by fitness, take top K for local refinement
        # We need to re-evaluate or track which points had what fitness
        # Since we only tracked the single best, let's do a simpler approach:
        # Run BFGS from the best point and a few perturbed variants
        
        def bfgs_local(x0, max_iter=50, tol=1e-8):
            """
            Accelerated BFGS-like local search using finite differences.
            Uses forward differences (cheaper) and adaptive step scaling.
            Returns (best_x, best_f) found during the search.
            """
            nonlocal best_f, best_x, calls_made
            
            if not budget_ok() or not time_ok():
                return x0.copy(), best_f
            
            x = x0.copy()
            fx = safe_call(x) if calls_made < total_budget else best_f
            if not np.isfinite(fx):
                return x0.copy(), best_f
            
            # Compute initial gradient via forward differences (cheaper: dim calls)
            h = 1e-4
            def compute_gradient(x_curr):
                g = np.zeros(dim)
                for d in range(dim):
                    xp = x_curr.copy()
                    xp[d] += h
                    xp = np.clip(xp, -5, 5)
                    if not budget_ok() or not time_ok():
                        return None
                    fp = safe_call(xp)
                    if not np.isfinite(fp):
                        return None
                    g[d] = (fp - fx) / h
                return g
            
            grad = compute_gradient(x)
            if grad is None:
                return x.copy(), fx
            
            # BFGS Hessian approximation
            H = np.eye(dim)
            step_scale = 0.5
            
            for it in range(max_iter):
                if not budget_ok() or not time_ok():
                    break
                
                # Search direction
                d = -H @ grad
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-10:
                    break
                
                d = d / d_norm * step_scale
                
                # Line search: try several step sizes, stop early if improvement found
                best_alpha = 0.5
                best_fx_line = fx
                for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                    x_trial = np.clip(x + alpha * d, -5, 5)
                    if not budget_ok() or not time_ok():
                        break
                    fx_trial = safe_call(x_trial)
                    if fx_trial < best_fx_line - 1e-10:
                        best_fx_line = fx_trial
                        best_alpha = alpha
                        # Early exit: good enough improvement
                        if fx_trial < fx - 0.01:
                            break
                
                x_new = np.clip(x + best_alpha * d, -5, 5)
                fx_new = best_fx_line
                
                if fx_new >= fx - 1e-10:
                    # No improvement, reduce step scale and try again
                    step_scale *= 0.5
                    if step_scale < 1e-4:
                        break
                    continue
                
                # Update gradient at new point
                grad_new = compute_gradient(x_new)
                if grad_new is None:
                    break
                
                # BFGS update
                s = x_new - x
                y = grad_new - grad
                sy = s @ y
                if sy > 1e-10:
                    Hy = H @ y
                    H = H - np.outer(Hy, y.T) / (y @ Hy) + np.outer(s, s.T) / sy
                    # Symmetrize and ensure positive definiteness
                    H = (H + H.T) / 2
                    # Regularize if needed
                    eigvals = np.linalg.eigvalsh(H)
                    if eigvals.min() < 1e-6:
                        H = H + (1e-5 - eigvals.min()) * np.eye(dim)
                
                x = x_new
                fx = fx_new
                grad = grad_new
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                if np.linalg.norm(grad) < tol:
                    break
                
                # Adaptive step scale: increase if we made good progress
                if fx_new < fx - 0.1:
                    step_scale = min(step_scale * 1.2, 1.0)
            
            return x.copy(), fx

        # Run BFGS from the best point with more iterations since gradient is cheaper now
        if budget_ok() and time_ok():
            bfgs_local(best_x.copy(), max_iter=40)
        
        # Run BFGS from several perturbed points around the best
        rng2 = np.random.default_rng(777)
        n_perturb = min(8, (total_budget - calls_made) // 15)
        for _ in range(n_perturb):
            if not budget_ok() or not time_ok():
                break
            # Use adaptive perturbation size
            pert_size = 0.3 + 0.2 * rng2.random()
            x_pert = best_x + pert_size * rng2.standard_normal(dim)
            x_pert = np.clip(x_pert, -5, 5)
            bfgs_local(x_pert, max_iter=25)
        
        # --- Phase 3: High-precision polish with tighter BFGS ---
        if best_x is not None and budget_ok() and time_ok():
            # Final polish: run BFGS again with very tight tolerance
            bfgs_local(best_x.copy(), max_iter=20, tol=1e-10)
        
        # --- Phase 4: Coordinate descent final touch (very cheap, 1 call per coord) ---
        if best_x is not None and budget_ok():
            x = best_x.copy()
            fx = best_f
            h_cd = 0.01
            for coord in range(dim):
                if not budget_ok() or not time_ok():
                    break
                # Try small moves in each direction
                for delta in [h_cd, -h_cd, h_cd/2, -h_cd/2, h_cd/10, -h_cd/10]:
                    if not budget_ok() or not time_ok():
                        break
                    x_trial = x.copy()
                    x_trial[coord] = np.clip(x_trial[coord] + delta, -5, 5)
                    fx_trial = safe_call(x_trial)
                    if np.isfinite(fx_trial) and fx_trial < fx - 1e-12:
                        x = x_trial
                        fx = fx_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
