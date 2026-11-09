import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Sampling to find a good starting point
        n_init = 20
        if budget < n_init:
            x0 = np.zeros(dim)
            fx0 = func(x0)
            return x0, fx0

        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        X_init[0] = 0.0
        
        F_init = np.array([func(x) for x in X_init])
        call_count = n_init
        
        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx].copy()
        f_best = F_init[best_idx]
        
        # Phase 2: Derivative-Free Trust-Region (DF-TR) Solver
        # We maintain a set of points (X) and their function values (F).
        # We fit a quadratic model to these points and use it to propose new points.
        
        # Initial trust region radius
        delta = 1.0 
        
        # Maximum trust region radius
        delta_max = 5.0 * np.sqrt(dim)
        
        # Minimum trust region radius
        delta_min = 1e-4
        
        # We keep a set of points. Size should be around (dim+1)(dim+2)/2 for a full quadratic model,
        # but for dim=5, that's 21 points. We can start with the initial points.
        # Let's maintain a list of points and values.
        X_set = [x_best]
        F_set = [f_best]
        
        # To fit a quadratic model, we need at least dim+1 points for linear, 
        # and (dim+1)(dim+2)/2 for quadratic.
        # For dim=5, we need 21 points for a unique quadratic fit (ignoring regularization).
        # We will add points until we have enough, then replace the worst ones or use regularization.
        
        # Let's use a simpler approach: 
        # 1. Start with x_best.
        # 2. Generate points on a sphere of radius delta around x_best.
        # 3. Evaluate them.
        # 4. Fit a quadratic model.
        # 5. Minimize the model to find the next step.
        
        # To ensure we have enough points for the model, let's first expand the set.
        # We need n_model = (dim+1)*(dim+2)//2 points.
        n_model = (dim + 1) * (dim + 2) // 2
        
        # Add more points to reach n_model if needed
        # We already have n_init points. If n_init >= n_model, we are good.
        # Otherwise, generate more points around x_best.
        
        # Actually, it's better to just use the points we have and add new ones as we go.
        # But fitting a model with too few points is ill-posed.
        # Let's force the initial set to be at least n_model points by sampling around x_best.
        
        # Clear and rebuild initial set for the model?
        # No, let's just keep the best point and generate a fresh local set.
        # This is safer for the model fit.
        
        # Reset set for local optimization
        X_set = [x_best]
        F_set = [f_best]
        
        # Generate n_model - 1 points around x_best
        # Use a random direction on the sphere
        for _ in range(n_model - 1):
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
            x_new = x_best + delta * direction
            # Clip to bounds?
            # If we clip, the model might be distorted.
            # Let's try without clipping first, or clip and accept.
            # For now, let's just generate and clip.
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = func(x_new)
            call_count += 1
            X_set.append(x_new)
            F_set.append(f_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
        
        # Now we have n_model points.
        # We will iterate:
        # 1. Fit quadratic model to X_set, F_set.
        # 2. Find minimizer of model within trust region.
        # 3. Evaluate at minimizer.
        # 4. Update trust region and set.
        
        # Quadratic Model: f(x) ~ f(x0) + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        # We fit this using least squares.
        
        def fit_quadratic_model(X, F, x0):
            # X: (N, dim), F: (N,), x0: (dim,)
            # We want to find g (dim) and H (dim x dim) such that:
            # F_i - F_0 ≈ g^T (X_i - x0) + 0.5 (X_i - x0)^T H (X_i - x0)
            # Let d_i = X_i - x0
            # Let y_i = F_i - F_0 (where F_0 is the value at x0, which might not be in X)
            # Wait, x0 is the center. We should use the value at x0 if available.
            # In our loop, x0 is the current best point, which is in X_set.
            # Let's find the index of x0 in X.
            
            # For simplicity, let's assume x0 is X[0] or we calculate F_0 separately?
            # No, we can just use the first point as the reference if we ensure x0 is the first point.
            # Or better, use the point in X closest to x0.
            # In our construction, X_set[0] is the initial x_best, but x_best might move.
            # Let's ensure we always keep the current center in the set.
            
            # Let's restructure:
            # The model is centered at x0.
            # We need F(x0). If x0 is not in X, we have to estimate it or use a different formulation.
            # Easier: The model is f(x) = c + g^T x + 0.5 x^T H x.
            # This requires fitting 1 + dim + dim*(dim+1)/2 parameters.
            # For dim=5, that's 1 + 5 + 15 = 21 parameters.
            # We have 21 points. This is a square system.
            
            # Let's use the full quadratic form: f(x) = c + g^T x + 0.5 x^T H x
            # Design matrix A:
            # Row i: [1, X_i, 0.5 * X_i^T H X_i terms]
            # The terms for 0.5 x^T H x are:
            # 0.5 * sum_{j,k} H_{jk} x_j x_k
            # = sum_{j<k} H_{jk} x_j x_k + 0.5 * sum_j H_{jj} x_j^2
            
            # So the features are:
            # 1
            # x_1, ..., x_d
            # x_1^2, x_2^2, ..., x_d^2
            # x_1 x_2, x_1 x_3, ..., x_{d-1} x_d
            
            # Number of features: 1 + d + d + d(d-1)/2 = 1 + 2d + d(d-1)/2 = (d^2 + 3d + 2)/2 = (d+1)(d+2)/2.
            # Matches n_model.
            
            def get_features(x):
                # x is (d,)
                # Returns (n_model,)
                feats = [1.0]
                feats.extend(x)
                for j in range(dim):
                    feats.append(x[j] * x[j])
                for j in range(dim):
                    for k in range(j + 1, dim):
                        feats.append(x[j] * x[k])
                return np.array(feats)
            
            A = np.array([get_features(x) for x in X])
            b = F
            
            # Solve A * coeffs = b
            # If A is ill-conditioned, use lstsq with regularization.
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                # Fallback to simple linear model or return None
                return None, None, None
            
            # coeffs: [c, g_1..g_d, H_11..H_dd, H_12..H_{d-1,d}]
            c = coeffs[0]
            g = coeffs[1 : 1 + dim]
            H_diag = coeffs[1 + dim : 1 + 2 * dim]
            H_off_diag = coeffs[1 + 2 * dim :]
            
            H = np.zeros((dim, dim))
            for j in range(dim):
                H[j, j] = H_diag[j]
            idx = 0
            for j in range(dim):
                for k in range(j + 1, dim):
                    H[j, k] = H_off_diag[idx]
                    H[k, j] = H_off_diag[idx]
                    idx += 1
            
            return c, g, H
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Fit model
            result = fit_quadratic_model(np.array(X_set), np.array(F_set), x_best)
            if result is None:
                break
            c, g, H = result
            
            # Find minimizer of the model within the trust region
            # Model: m(x) = c + g^T x + 0.5 x^T H x
            # Gradient: g + H x
            # Hessian: H
            # We want to minimize m(x) subject to ||x - x_best|| <= delta
            
            # If H is positive definite, the unconstrained minimizer is x* = -H^{-1} g
            # If H is not positive definite, the minimizer is on the boundary.
            
            # Let's check eigenvalues of H
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Regularize H if necessary
            # If min eigval < 0, we have a saddle point or max.
            # For a sphere function, H should be PD.
            # If not, we might be far from the minimum or the model is bad.
            
            # Simple approach: use scipy.optimize.minimize on the model?
            # No, let's do it analytically or with a simple solver.
            # Since dim=5, we can use scipy.optimize.minimize with trust-constr?
            # Or just use the unconstrained minimizer if it's inside the region.
            
            # Check if H is PD
            if np.min(eigvals) > 1e-8:
                # Unconstrained minimizer
                try:
                    x_uncon = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    x_uncon = x_best
            
                # Check if inside trust region
                dist = np.linalg.norm(x_uncon - x_best)
                if dist <= delta:
                    x_trial = x_uncon
                else:
                    # Project onto trust region boundary
                    # x_trial = x_best + delta * (x_uncon - x_best) / dist
                    x_trial = x_best + delta * (x_uncon - x_best) / dist
            else:
                # H is not PD. 
                # This can happen if we are far away or the model is bad.
                # Let's just take a step in the negative gradient direction?
                # Or use the boundary point in the direction of -g?
                # For now, let's just move slightly in the -g direction if g is not zero.
                if np.linalg.norm(g) > 1e-8:
                    x_trial = x_best - delta * g / np.linalg.norm(g)
                else:
                    x_trial = x_best.copy()
            
            # Clip to bounds
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            # Evaluate
            f_trial = func(x_trial)
            call_count += 1
            
            # Update trust region
            actual_reduction = f_best - f_trial
            predicted_reduction = (c + g.T @ x_best + 0.5 * x_best.T @ H @ x_best) - (c + g.T @ x_trial + 0.5 * x_trial.T @ H @ x_trial)
            
            # Note: The predicted reduction calculation above is approximate if x_best is not exactly the center of the model in the way we derived it?
            # Actually, the model is a global fit, so m(x_best) is the model value at x_best.
            # m(x_best) = c + g.T @ x_best + 0.5 * x_best.T @ H @ x_best
            # m(x_trial) = c + g.T @ x_trial + 0.5 * x_trial.T @ H @ x_trial
            # So predicted_reduction = m(x_best) - m(x_trial)
            
            if actual_reduction > 0:
                # Success
                if actual_reduction / max(abs(predicted_reduction), 1e-8) > 0.75:
                    delta = min(2 * delta, delta_max)
                elif actual_reduction / max(abs(predicted_reduction), 1e-8) > 0.25:
                    # Keep delta
                    pass
                else:
                    delta = delta / 2.0
            else:
                # Failure
                delta = delta / 2.0
            
            # Update best
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial.copy()
            
            # Update X_set and F_set
            # We want to keep the points that are most informative.
            # Simple strategy: Replace the worst point (highest function value) with the new point?
            # Or replace the point furthest from the center?
            # Let's replace the point with the highest function value to keep the set focused on the minimum.
            # But we need to ensure we don't lose diversity.
            
            # Let's just append and if len > n_model, remove the one with the highest F.
            X_set.append(x_trial)
            F_set.append(f_trial)
            
            if len(X_set) > n_model:
                # Find index of max F
                idx_max = np.argmax(F_set)
                X_set.pop(idx_max)
                F_set.pop(idx_max)
            
            # Stopping criteria
            if delta < delta_min:
                break
            if np.linalg.norm(g) < 1e-6:
                break

        return x_best, f_best
