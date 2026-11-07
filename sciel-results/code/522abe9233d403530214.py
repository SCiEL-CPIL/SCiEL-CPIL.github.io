"""Combined optimizer: Quadratic model fitting for global localization, followed by Hessian-scaled (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that leverages the known quadratic structure of the Sphere function.
    
    Strategy:
    1. Sample a set of well-conditioned points to estimate the quadratic coefficients.
    2. Solve for the stationary point (minimum) of the fitted quadratic.
    3. Use this point as the initial guess for a Hessian-scaled (1+1)-ES local search
       with adaptive sigma via the 1/5 success rule.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def make_call(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline - 2.0:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Quadratic Model Fitting ---
        # For Sphere: f(x) = (x - x*)^T (x - x*) + f_opt
        # We fit: f(x) = x^T A x + b^T x + c
        # Design points: center + symmetric axis pairs + symmetric cross pairs
        
        # Generate design points
        design = []
        design.append(np.zeros(n))  # center
        
        scale = 3.0
        
        # Axis points: ±scale along each axis (2*n points)
        for i in range(n):
            for s in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = s * scale
                design.append(pt)
        
        # Cross points: ±scale/sqrt(2) along pairs (2 * n*(n-1)/2 points)
        for i in range(n):
            for j in range(i + 1, n):
                for s in [-1, 1]:
                    pt = np.zeros(n)
                    pt[i] = s * scale / np.sqrt(2)
                    pt[j] = s * scale / np.sqrt(2)
                    design.append(pt)
        
        # We need at most n*(n+1)/2 + n + 1 = 21 points for n=5
        # But our design has 1 + 2*n + 2*n*(n-1)/2 = 1 + 2n + n(n-1) = 1 + n^2 + n points
        # For n=5: 1 + 25 + 5 = 31 points. We'll use all of them for robustness.
        
        n_design = len(design)
        X = np.array(design)
        F = np.zeros(n_design)
        
        for i in range(n_design):
            fx = make_call(X[i])
            if fx is None:
                break
            F[i] = fx
        
        if calls < 2 or best_x is None:
            # Fallback: random search
            for _ in range(budget - calls):
                if time.time() > deadline - 2.0:
                    break
                x = rng.uniform(-5, 5, n)
                fx = make_call(x)
                if fx is None:
                    break
            return best_x, best_f
        
        # Fit quadratic model
        # Features: [x_i*x_j for i<=j, x_i for i, 1]
        n_quad = n * (n + 1) // 2
        n_params = n_quad + n + 1
        
        def quad_features(x):
            feats = np.zeros(n_params)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    feats[idx] = x[i] * x[j]
                    idx += 1
            for i in range(n):
                feats[idx] = x[i]
                idx += 1
            feats[idx] = 1.0
            return feats
        
        M = np.array([quad_features(x) for x in X[:calls]])
        F_valid = F[:calls]
        
        # Solve least squares
        try:
            params, residuals, rank, sv = np.linalg.lstsq(M, F_valid, rcond=None)
        except np.linalg.LinAlgError:
            params = np.zeros(n_params)
        
        # Extract A, b, c from params
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                val = params[idx]
                A[i, j] = val
                if i != j:
                    A[j, i] = val
                idx += 1
        
        b = params[n_quad:n_quad + n]
        c = params[-1]
        
        # Solve for x*: minimize x^T A x + b^T x + c
        # Gradient: 2Ax + b = 0 => x* = -0.5 * A^{-1} b
        try:
            # Regularize for numerical stability
            A_reg = A + 1e-10 * np.eye(n)
            x_star = -0.5 * np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            try:
                x_star = -0.5 * np.linalg.lstsq(A + 1e-10 * np.eye(n), b, rcond=None)[0]
            except:
                x_star = np.zeros(n)
        
        x_star = np.clip(x_star, -5, 5)
        
        # Evaluate at the estimated minimum
        fx_star = make_call(x_star)
        if fx_star is None:
            return best_x, best_f
        
        # --- Phase 2: Hessian-scaled (1+1)-ES Polish ---
        # Use eigenvalues of A for adaptive step scaling
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.clip(eigvals, 1e-10, 1e10)
            # Step scale per dimension: 1/sqrt(2*lambda_i)
            step_scale = 1.0 / np.sqrt(2.0 * eigvals)
            # Normalize so max scale is 1
            step_scale = step_scale / np.max(step_scale)
        except:
            step_scale = np.ones(n) / np.sqrt(n)
        
        # Initialize (1+1)-ES
        x_cur = x_star.copy()
        f_cur = fx_star
        
        # Start with moderate sigma - the quadratic fit should be very close
        sigma = 0.5
        sigma_min = 1e-14
        sigma_max = 5.0
        
        # 1/5 success rule
        window = 10
        successes = 0
        sigma_up = 1.05
        sigma_down = 0.95
        
        # Restart logic
        no_progress = 0
        restart_threshold = 30
        max_restarts = 5
        restarts = 0
        
        while calls < budget:
            if time.time() > deadline - 2.0:
                break
            
            # Generate perturbation
            z = rng.standard_normal(n)
            delta = sigma * step_scale * z
            x_new = x_cur + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = make_call(x_new)
            if f_new is None:
                break
            
            # 1/5 rule
            if f_new < f_cur:
                x_cur = x_new.copy()
                f_cur = f_new
                successes += 1
                no_progress = 0
            else:
                no_progress += 1
            
            # Adapt sigma every window steps
            if (calls % window) == 0:
                if successes / window > 0.2:
                    sigma *= sigma_up
                else:
                    sigma *= sigma_down
                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = 0
            
            # Restart if stuck
            if no_progress >= restart_threshold and restarts < max_restarts:
                restarts += 1
                no_progress = 0
                x_cur = best_x.copy()
                f_cur = best_f
                sigma = 0.1  # Smaller sigma for restart
        
        return best_x, best_f
# EVOLVE-BLOCK-END
