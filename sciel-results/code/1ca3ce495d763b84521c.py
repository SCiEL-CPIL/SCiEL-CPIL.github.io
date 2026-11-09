import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        call_count = 0

        # Phase 1: Global Sampling to find a good starting point
        n_init = 25
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
        delta_min = 1e-6

        # We keep a set of points. Size should be around (dim+1)(dim+2)/2 for a full quadratic model,
        # but for dim=5, that's 21 points.
        n_model = (dim + 1) * (dim + 2) // 2

        # Initial set for the model: start with the best point and generate points around it
        X_set = [x_best]
        F_set = [f_best]

        # Generate n_model - 1 points around x_best using random directions
        for _ in range(n_model - 1):
            direction = np.random.randn(dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            else:
                direction = np.zeros(dim)
                direction[0] = 1.0
            x_new = x_best + delta * direction
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = func(x_new)
            call_count += 1
            X_set.append(x_new)
            F_set.append(f_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()

        # Function to fit a quadratic model centered at x0
        def fit_quadratic_model(X, F, x0):
            # X: (N, dim), F: (N,), x0: (dim,)
            # Model: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
            # We need f0 = F(x0). If x0 is in X, we can use that value.
            # Otherwise, we estimate it or use a different formulation.
            # For simplicity, let's assume x0 is the first point in X or we find the closest.
            # In our construction, x_best is always in X_set.
            
            # Find index of x0 in X
            dists = np.linalg.norm(X - x0, axis=1)
            idx0 = np.argmin(dists)
            f0 = F[idx0]
            
            # If x0 is not exactly in X, f0 might be slightly off, but it's the best estimate.
            # Actually, for the model to be accurate, we should use the value at x0.
            # Let's ensure x0 is in X by construction.
            
            # Features: d, d^2, d*d
            # d_i = X_i - x0
            D = X - x0
            N = len(X)
            
            # Design matrix A: N x (1 + dim + dim + dim*(dim-1)/2)
            # Columns:
            # 1: constant (for f0, but we subtract f0 so this is 0? No, we fit f(x) - f0)
            # Actually, let's fit: y_i = F_i - f0
            # y_i = g^T d_i + 0.5 d_i^T H d_i
            # Features for d_i:
            # d_i[0], ..., d_i[dim-1]  (dim features)
            # d_i[0]^2, ..., d_i[dim-1]^2 (dim features)
            # d_i[0]*d_i[1], d_i[0]*d_i[2], ..., d_i[dim-2]*d_i[dim-1] (dim*(dim-1)/2 features)
            
            n_features = dim + dim + dim * (dim - 1) // 2
            A = np.zeros((N, n_features))
            
            for i in range(N):
                d = D[i]
                col = 0
                # Linear terms
                for j in range(dim):
                    A[i, col] = d[j]
                    col += 1
                # Quadratic diagonal terms
                for j in range(dim):
                    A[i, col] = d[j] * d[j]
                    col += 1
                # Quadratic off-diagonal terms
                for j in range(dim):
                    for k in range(j + 1, dim):
                        A[i, col] = d[j] * d[k]
                        col += 1
            
            y = F - f0
            
            # Solve A * coeffs = y
            # coeffs: [g_1..g_d, H_11..H_dd, H_12..H_{d-1,d}]
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            except:
                return None, None, None, f0
            
            g = coeffs[0:dim].copy()
            H_diag = coeffs[dim:2*dim].copy()
            H_off = coeffs[2*dim:].copy()
            
            H = np.zeros((dim, dim))
            for j in range(dim):
                H[j, j] = H_diag[j]
            idx = 0
            for j in range(dim):
                for k in range(j + 1, dim):
                    H[j, k] = H_off[idx]
                    H[k, j] = H_off[idx]
                    idx += 1
            
            return g, H, f0, f0

        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Fit model
            result = fit_quadratic_model(np.array(X_set), np.array(F_set), x_best)
            if result is None:
                break
            g, H, f0, _ = result
            
            # Find minimizer of the model within the trust region
            # Model: m(x) = f0 + g^T (x - x_best) + 0.5 (x - x_best)^T H (x - x_best)
            # Gradient: g + H (x - x_best)
            # Hessian: H
            # Unconstrained minimizer: x* = x_best - H^{-1} g
            
            # Check if H is positive definite
            eigvals, eigvecs = np.linalg.eigh(H)
            
            if np.min(eigvals) > 1e-8:
                # H is PD
                try:
                    step = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    step = -g / np.linalg.norm(g) if np.linalg.norm(g) > 1e-8 else np.zeros(dim)
                
                dist = np.linalg.norm(step)
                if dist > delta:
                    step = step * (delta / dist)
            else:
                # H is not PD. Use gradient descent step.
                if np.linalg.norm(g) > 1e-8:
                    step = -delta * g / np.linalg.norm(g)
                else:
                    step = np.zeros(dim)
            
            x_trial = x_best + step
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            # Evaluate
            f_trial = func(x_trial)
            call_count += 1
            
            # Update trust region
            # Predicted reduction
            d = x_trial - x_best
            pred_red = -(g.T @ d + 0.5 * d.T @ H @ d)
            actual_red = f_best - f_trial
            
            if actual_red > 0:
                # Success
                if abs(pred_red) > 1e-8:
                    rho = actual_red / pred_red
                    if rho > 0.75:
                        delta = min(2 * delta, delta_max)
                    elif rho > 0.25:
                        pass
                    else:
                        delta = delta / 2.0
                else:
                    delta = min(2 * delta, delta_max)
            else:
                # Failure
                delta = delta / 2.0
            
            # Update best
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial.copy()
            
            # Update X_set and F_set
            # Replace the worst point (highest F) with the new point
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

        # Phase 3: Final Polish with Gradient Descent
        # Use finite differences to estimate gradient and take small steps
        if call_count < budget:
            eps = 1e-4
            x_curr = x_best.copy()
            f_curr = f_best
            
            # Estimate gradient
            for _ in range(10):  # Max 10 polish steps
                if call_count >= budget or time.time() > deadline:
                    break
                
                grad = np.zeros(dim)
                for i in range(dim):
                    if call_count + 2 > budget or time.time() > deadline:
                        break
                    x_plus = x_curr.copy()
                    x_plus[i] += eps
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = x_curr.copy()
                    x_minus[i] -= eps
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    f_plus = func(x_plus)
                    call_count += 1
                    f_minus = func(x_minus)
                    call_count += 1
                    
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-8:
                    break
                
                # Take a step
                alpha = 1.0
                improved = False
                for _ in range(10):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x_new = x_curr - alpha * grad
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = func(x_new)
                    call_count += 1
                    
                    if f_new < f_curr:
                        x_curr = x_new
                        f_curr = f_new
                        improved = True
                        break
                    alpha *= 0.5
                
                if not improved:
                    break
                
                if f_curr < f_best:
                    f_best = f_curr
                    x_best = x_curr.copy()

        return x_best, f_best
