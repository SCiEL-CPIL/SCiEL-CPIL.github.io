# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region solver with local quadratic surrogate model."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        x_best = np.zeros(n)
        f_best = np.inf
        calls = 0

        # Trust region
        rho = 1.0
        rho_min = 1e-6
        rho_max = 8.0

        # Model parameters
        c_model = 0.0
        g_model = np.zeros(n)
        H_model = np.zeros((n, n))
        model_ok = False

        # Helper to evaluate
        def evaluate(x):
            nonlocal x_best, f_best, calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e6
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return x, fx

        # Fit quadratic model using points in X, centered at x0
        def do_fit(x0, X_list, F_list):
            nonlocal c_model, g_model, H_model, model_ok
            X = np.array(X_list)
            F = np.array(F_list)
            m = len(X)
            if m < n + 1:
                model_ok = False
                return
            
            # Select points closest to x0 to ensure local validity
            dists = np.linalg.norm(X - x0, axis=1)
            # We want enough points. Let's take up to 2*(n+1) points if possible, or all if fewer.
            # For quadratic, we ideally want ~ n^2 points.
            max_pts = min(m, max(2*n, n*(n+1)//2 + 5))
            idx = np.argsort(dists)[:max_pts]
            Xs = X[idx]
            Fs = F[idx]
            
            dx = Xs - x0
            # Design matrix: [1, dx, dx^2]
            # Quadratic terms: dx_i * dx_j for i <= j
            quad_cols = []
            for i in range(n):
                for j in range(i, n):
                    quad_cols.append(dx[:, i] * dx[:, j])
            
            A = np.hstack([np.ones((len(Xs), 1)), dx, np.column_stack(quad_cols)])
            b = Fs
            
            try:
                coeffs, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
                # Check rank
                if rank < A.shape[1] - 1: # Allow some deficiency
                    model_ok = False
                    return
            except:
                model_ok = False
                return

            c_model = coeffs[0]
            g_model = coeffs[1:1+n].copy()
            
            H = np.zeros((n, n))
            k = 1 + n
            for i in range(n):
                for j in range(i, n):
                    val = coeffs[k]
                    if i == j:
                        H[i, i] += 2.0 * val
                    else:
                        H[i, j] += val
                        H[j, i] += val
                    k += 1
            H_model = H
            model_ok = True

        # Solve trust region subproblem
        def solve_tr(x0, rho):
            g = g_model
            H = H_model
            
            # Check if H is positive definite
            try:
                eigvals = np.linalg.eigvalsh(H)
                is_pd = np.min(eigvals) > 1e-8
            except:
                is_pd = False

            if is_pd:
                # Newton step
                try:
                    d_newton = -np.linalg.solve(H, g)
                except:
                    d_newton = -g / (np.linalg.norm(g) + 1e-12)
                norm_d = np.linalg.norm(d_newton)
                if norm_d > rho:
                    d_newton = d_newton * (rho / norm_d)
                return x0 + d_newton
            else:
                # Steepest descent
                norm_g = np.linalg.norm(g)
                if norm_g < 1e-12:
                    return x0
                d_sd = -g / norm_g
                return x0 + rho * d_sd

        # Main Loop
        # 1. Initialization
        x_cur = rng.uniform(-5, 5, n)
        x_cur, f_cur = evaluate(x_cur)
        
        # Build initial interpolation set
        pool_X = [x_cur]
        pool_F = [f_cur]
        
        # Generate initial neighborhood
        init_radius = 0.5
        # Coordinate points
        for i in range(n):
            for sign in [1, -1]:
                dx = np.zeros(n)
                dx[i] = sign * init_radius
                x_new = x_cur + dx
                x_new, f_new = evaluate(x_new)
                pool_X.append(x_new)
                pool_F.append(f_new)
                
        # Random points to complete the set
        # We need at least n*(n+1)/2 + n + 1 points for a unique quadratic fit
        # For n=5, that is 15 + 5 + 1 = 21 points.
        # We have 1 (center) + 10 (coord) = 11 points.
        # Need 10 more.
        for _ in range(10):
            dx = rng.standard_normal(n) * init_radius
            x_new = x_cur + dx
            x_new, f_new = evaluate(x_new)
            pool_X.append(x_new)
            pool_F.append(f_new)

        # Fit model
        do_fit(x_cur, pool_X, pool_F)

        # 2. Iterative Optimization
        while calls < budget:
            if not model_ok:
                # Fallback: random step
                dx = rng.standard_normal(n) * rho
                x_cand = x_best + dx
                x_cand, f_cand = evaluate(x_cand)
                pool_X.append(x_cand)
                pool_F.append(f_cand)
                # Try to fit again
                do_fit(x_best, pool_X, pool_F)
                if not model_ok:
                    rho = max(rho * 0.5, rho_min)
                    continue
            
            # Solve TR subproblem
            x_cand = solve_tr(x_best, rho)
            x_cand, f_cand = evaluate(x_cand)
            
            # Add to pool
            pool_X.append(x_cand)
            pool_F.append(f_cand)
            
            # Update Trust Region
            # Predicted reduction
            dx = x_cand - x_best
            pred_red = -(g_model @ dx + 0.5 * dx @ H_model @ dx)
            act_red = f_best - f_cand
            
            if pred_red > 1e-12:
                ratio = act_red / pred_red
            else:
                ratio = 1.0 if act_red > 0 else 0.0

            if ratio > 0.75:
                rho = min(rho * 2.0, rho_max)
                # Move center to candidate
                x_cur = x_cand.copy()
            elif ratio > 0.25:
                # Accept step but don't expand
                x_cur = x_cand.copy()
            else:
                rho = max(rho * 0.5, rho_min)
                # Don't move center
            
            # Refit model at current center (x_best or x_cur?)
            # Usually refit at the current best point to ensure model is local to the best.
            do_fit(x_best, pool_X, pool_F)
            
            # If model is bad, add more points
            if not model_ok and calls < budget:
                for _ in range(5):
                    if calls >= budget: break
                    dx = rng.standard_normal(n) * rho
                    x_new = x_best + dx
                    x_new, f_new = evaluate(x_new)
                    pool_X.append(x_new)
                    pool_F.append(f_new)
                do_fit(x_best, pool_X, pool_F)

        return x_best, f_best

# EVOLVE-BLOCK-END
