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

        # Phase 1: Improved Global Sampling with LHS for better coverage
        n_init = 25
        if budget < n_init:
            x0 = np.zeros(dim)
            fx0 = func(x0)
            return x0, fx0

        # Generate initial candidates: LHS + origin + Gaussian
        X_init = np.zeros((n_init, dim))
        # First point: origin
        X_init[0] = 0.0
        
        # Latin Hypercube Sampling for next 12 points (better space-filling)
        for i in range(1, 13):
            # Each dimension gets one sample per stratum
            for d in range(dim):
                # Random position within stratum i
                X_init[i, d] = -5.0 + (i + np.random.random()) * (10.0 / 12.0)
        
        # Next 8: Gaussian centered at 0 with sigma=2.5
        for i in range(13, 21):
            X_init[i] = np.clip(np.random.normal(0, 2.5, dim), -5, 5)
        
        # Last 4: Wider Gaussian for exploration
        for i in range(21, 25):
            X_init[i] = np.clip(np.random.normal(0, 4.0, dim), -5, 5)

        F_init = np.array([func(x) for x in X_init])
        call_count = n_init

        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx].copy()
        f_best = F_init[best_idx]

        # Phase 2: Derivative-Free Trust-Region (DF-TR) Solver
        # We maintain a set of points (X) and their function values (F).
        # We fit a quadratic model to these points and use it to propose new points.

        # Initial trust region radius - start larger for faster initial progress
        delta = 2.0

        # Maximum trust region radius
        delta_max = 5.0 * np.sqrt(dim)

        # Minimum trust region radius
        delta_min = 1e-6

        # We keep a set of points. Size should be around (dim+1)(dim+2)/2 for a full quadratic model,
        # but for dim=5, that's 21 points.
        n_model = (dim + 1) * (dim + 2) // 2

        # Initial set for the model: start with the best point and generate points around it
        # Use structured directions (coordinate + random) for better model quality
        X_set = [x_best]
        F_set = [f_best]

        # Generate n_model - 1 points around x_best
        # First: coordinate directions (forward and backward)
        n_coords = 2 * dim
        n_random = n_model - 1 - n_coords
        if n_random < 0:
            n_random = 0
        
        for d in range(dim):
            for sign in [1.0, -1.0]:
                if len(X_set) >= n_model:
                    break
                direction = np.zeros(dim)
                direction[d] = sign
                x_new = x_best + delta * direction
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                call_count += 1
                X_set.append(x_new)
                F_set.append(f_new)
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
        
        # Then: random directions
        for _ in range(n_random):
            if len(X_set) >= n_model:
                break
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
            
            # Find index of x0 in X
            dists = np.linalg.norm(X - x0, axis=1)
            idx0 = np.argmin(dists)
            f0 = F[idx0]
            
            D = X - x0
            N = len(X)
            
            # Features: d, d^2, d*d
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
            d = x_trial - x_best
            pred_red = -(g.T @ d + 0.5 * d.T @ H @ d)
            actual_red = f_best - f_trial
            
            if actual_red > 0:
                # Success
                if abs(pred_red) > 1e-8:
                    rho = actual_red / pred_red
                    if rho > 0.75:
                        delta = min(2.5 * delta, delta_max)  # More aggressive expansion
                    elif rho > 0.25:
                        delta = min(1.2 * delta, delta_max)  # Slight expansion
                    else:
                        delta = delta / 2.0
                else:
                    delta = min(2.0 * delta, delta_max)
            else:
                # Failure
                delta = delta / 2.0
            
            # Update best
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial.copy()
            
            # Update X_set and F_set
            # Strategy: Replace the point that is farthest from x_best (to keep model local)
            X_set.append(x_trial)
            F_set.append(f_trial)
            
            if len(X_set) > n_model:
                # Find the point farthest from x_best
                X_arr = np.array(X_set)
                dists_from_best = np.linalg.norm(X_arr - x_best, axis=1)
                idx_farthest = np.argmax(dists_from_best)
                
                # Remove the farthest point to keep the model local
                X_set.pop(idx_farthest)
                F_set.pop(idx_farthest)
            
            # Stopping criteria - be more lenient to allow more iterations
            if delta < delta_min:
                break
            if np.linalg.norm(g) < 1e-8:
                break

        # Phase 3: Final Polish using the quadratic model
        # Try to take more Newton steps with finer damping for high-precision convergence
        if call_count < budget:
            polish_iters = min(10, budget - call_count)
            for polish_iter in range(polish_iters):
                if call_count >= budget or time.time() > deadline:
                    break
                
                result = fit_quadratic_model(np.array(X_set), np.array(F_set), x_best)
                if result is None:
                    break
                g, H, f0, _ = result
                
                eigvals, eigvecs = np.linalg.eigh(H)
                
                if np.min(eigvals) > 1e-10:
                    try:
                        step = np.linalg.solve(H, -g)
                    except np.linalg.LinAlgError:
                        break
                    
                    # Take a damped step with more damping options for precision
                    found_improvement = False
                    for damp in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                        if call_count >= budget:
                            break
                        x_trial = x_best + damp * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = func(x_trial)
                        call_count += 1
                        
                        if f_trial < f_best:
                            f_best = f_trial
                            x_best = x_trial.copy()
                            X_set.append(x_trial)
                            F_set.append(f_trial)
                            if len(X_set) > n_model:
                                X_arr = np.array(X_set)
                                dists_from_best = np.linalg.norm(X_arr - x_best, axis=1)
                                idx_farthest = np.argmax(dists_from_best)
                                X_set.pop(idx_farthest)
                                F_set.pop(idx_farthest)
                            found_improvement = True
                            break
                    
                    if not found_improvement:
                        break

        return x_best, f_best
