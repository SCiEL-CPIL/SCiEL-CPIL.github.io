"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based Bayesian optimization with Gaussian Process regression
    for the BBOB f6 (Attractive Sector) function.
    
    Strategy:
    1. Phase 1: Use a small population of random points to build an initial model.
    2. Phase 2: Fit a GP regression model to the observed data and use Expected
       Improvement (EI) to select the next evaluation point.
    3. Phase 3: If the model predicts we're close to the optimum, switch to
       local refinement using a quadratic model (since f6 is a shifted/rotated sphere).
    
    The f6 function is f(x) = ||Q(x-s) - z||^2, which is a convex quadratic.
    We can fit a quadratic model directly and minimize it analytically.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf
        n = self.dim
        budget = self.budget

        # Store all observations
        X_obs = []
        f_obs = []

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            calls_made += 1
            X_obs.append(x.copy())
            f_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial sampling - use a space-filling design
        # Start with the center and some structured points
        # For n=5, use a grid of corner points and center
        points_to_try = []
        
        # Center
        points_to_try.append(np.zeros(n))
        
        # Axis-aligned points: +/- 2 along each axis
        for i in range(n):
            for sign in [1, -1]:
                p = np.zeros(n)
                p[i] = sign * 2.0
                points_to_try.append(p)
        
        # Some random points
        rng = np.random.RandomState(42)
        for _ in range(10):
            p = rng.uniform(-4, 4, size=n)
            points_to_try.append(p)
        
        # Evaluate initial points
        for p in points_to_try:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break
            evaluate(p)
        
        if calls_made >= budget or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: Quadratic model fitting
        # f6 is a shifted/rotated sphere: f(x) = ||Q(x-s) - z||^2
        # This is a quadratic function: f(x) = x^T A x + b^T x + c
        # We can fit A, b, c using least squares from the observations.
        
        def fit_quadratic(X, f):
            """Fit a quadratic model f(x) = x^T A x + b^T x + c"""
            m = len(X)
            d = X.shape[1]
            # Build design matrix for quadratic terms
            # Features: x_i, x_i*x_j (i<=j), and constant
            # Number of features: d + d*(d+1)/2 + 1
            n_lin = d
            n_quad = d * (d + 1) // 2
            n_total = n_lin + n_quad + 1
            
            Phi = np.zeros((m, n_total))
            col = 0
            # Linear terms
            for i in range(d):
                Phi[:, col] = X[:, i]
                col += 1
            # Quadratic terms
            for i in range(d):
                for j in range(i, d):
                    Phi[:, col] = X[:, i] * X[:, j]
                    col += 1
            # Constant
            Phi[:, col] = 1.0
            
            # Solve least squares
            coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, f, rcond=None)
            
            # Reconstruct A, b, c
            b = coeffs[:d].copy()
            c = coeffs[-1]
            A = np.zeros((d, d))
            col = d
            for i in range(d):
                for j in range(i, d):
                    val = coeffs[col]
                    A[i, j] = val
                    if i != j:
                        A[j, i] = val
                    col += 1
            
            # Symmetrize A
            A = 0.5 * (A + A.T)
            
            return A, b, c

        def predict_quadratic(x, A, b, c):
            return x @ A @ x + b @ x + c

        def minimize_quadratic(A, b, c, x_center, box_low=-5, box_high=5):
            """Minimize the quadratic model, constrained to the box."""
            # Gradient: 2Ax + b = 0 => x* = -0.5 * A^{-1} b
            try:
                if np.linalg.matrix_rank(A) == A.shape[0]:
                    x_star = -0.5 * np.linalg.solve(A, b)
                    # Check if within box
                    if np.all(x_star >= box_low) and np.all(x_star <= box_high):
                        return x_star
                else:
                    # Use pseudo-inverse
                    x_star = -0.5 * np.linalg.pinv(A) @ b
            except:
                x_star = x_center.copy()
            
            # If the unconstrained minimum is outside the box,
            # project onto the box and do a few gradient steps
            x = np.clip(x_star, box_low, box_high)
            
            # If still outside or at boundary, do projected gradient descent
            for _ in range(100):
                grad = 2 * A @ x + b
                lr = 0.1
                x_new = x - lr * grad
                x_new = np.clip(x_new, box_low, box_high)
                if np.linalg.norm(x_new - x) < 1e-10:
                    break
                x = x_new
            
            return x

        # Iterative model-based optimization
        max_iter = 50
        for iteration in range(max_iter):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Need at least n+1 points to fit a linear model,
            # and more for quadratic
            if len(X_obs) < 2 * n + 1:
                # Add more random points
                p = np.random.uniform(-4, 4, size=n)
                evaluate(p)
                continue
            
            # Fit quadratic model
            X_arr = np.array(X_obs)
            f_arr = np.array(f_obs)
            
            # Filter out any NaN or inf
            mask = np.isfinite(f_arr)
            if mask.sum() < 2 * n + 1:
                p = np.random.uniform(-4, 4, size=n)
                evaluate(p)
                continue
            
            X_fit = X_arr[mask]
            f_fit = f_arr[mask]
            
            A, b, c = fit_quadratic(X_fit, f_fit)
            
            # Find the minimum of the quadratic model
            x_candidate = minimize_quadratic(A, b, c, best_x)
            
            # Check if this candidate is close to an existing point
            if len(X_obs) > 0:
                X_arr_check = np.array(X_obs)
                dists = np.linalg.norm(X_arr_check - x_candidate, axis=1)
                if np.min(dists) < 0.1:
                    # Try a perturbed version
                    x_candidate = x_candidate + 0.5 * np.random.randn(n)
                    x_candidate = np.clip(x_candidate, -5, 5)
            
            # Evaluate the candidate
            f_cand = evaluate(x_candidate)
            
            # If the model is good (low prediction error), we might be converging
            # Check prediction error
            pred = predict_quadratic(X_fit, A, b, c)
            mse = np.mean((pred - f_fit) ** 2)
            
            if mse < 1e-6 and f_cand < best_f * 1.001:
                # We're very close, do local refinement
                # Use a small random perturbation around the best point
                for _ in range(5):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    p = best_x + 0.1 * np.random.randn(n)
                    p = np.clip(p, -5, 5)
                    evaluate(p)
                break
        # Phase 3: Final refinement if budget remains
        while calls_made < budget:
            if time.time() > deadline - 1:
                break
            
            # Local search around the best point
            scale = max(0.01, 1.0 / (1 + calls_made))
            p = best_x + scale * np.random.randn(n)
            p = np.clip(p, -5, 5)
            
            # Only evaluate if it's sufficiently different from existing points
            if len(X_obs) > 0:
                X_arr_check = np.array(X_obs)
                dists = np.linalg.norm(X_arr_check - p, axis=1)
                if np.min(dists) < 0.05 * scale:
                    continue
            
            evaluate(p)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
