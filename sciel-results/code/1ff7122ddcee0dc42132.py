"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize with a diverse set of points to build a good initial model
        # We need at least n*(n+1)/2 + 1 points for a full quadratic fit.
        # For n=5, that's 16 points.
        min_points_for_fit = n * (n + 1) // 2 + 1
        
        history = []
        
        # 1. Evaluate origin (or center)
        x0 = np.zeros(n)
        f0 = func(x0)
        history.append((x0, f0))
        
        # 2. Evaluate random points
        for _ in range(3):
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            history.append((x, fx))
            
        # 3. Evaluate axis-aligned perturbations from the best point found so far
        # This helps capture the gradient direction early.
        x_best = min(history, key=lambda h: h[1])[0]
        f_best = min(f for _, f in history)
        
        step_sizes = [1.0, 0.5, 2.0]
        for i in range(n):
            for s in step_sizes:
                x_plus = x_best.copy()
                x_plus[i] += s
                x_minus = x_best.copy()
                x_minus[i] -= s
                # Only evaluate if within bounds
                if np.all(x_plus >= -5) and np.all(x_plus <= 5):
                    fp = func(x_plus)
                    history.append((x_plus, fp))
                    if fp < f_best:
                        f_best = fp
                        x_best = x_plus.copy()
                if np.all(x_minus >= -5) and np.all(x_minus <= 5):
                    fm = func(x_minus)
                    history.append((x_minus, fm))
                    if fm < f_best:
                        f_best = fm
                        x_best = x_minus.copy()

        call_count = len(history)
        
        # Adaptive step size for random fallback
        sigma = 1.0
        max_sigma = 5.0
        min_sigma = 1e-3
        
        while call_count < budget:
            # If we have enough points, try to fit a local quadratic model
            if len(history) >= min_points_for_fit:
                # Select local points: closest to x_best
                # We take the 'min_points_for_fit' closest points to x_best
                # This ensures the model is locally valid
                dists = [np.linalg.norm(h[0] - x_best) for h in history]
                # Get indices of the closest points
                closest_indices = np.argsort(dists)[:min_points_for_fit]
                
                X = np.array([history[i][0] for i in closest_indices])
                F = np.array([history[i][1] for i in closest_indices])
                
                # Build design matrix for quadratic fit: [1, x, x_i*x_j (i<=j)]
                m = n * (n + 1) // 2
                Q = np.zeros((len(X), 1 + n + m))
                for i, xi in enumerate(X):
                    Q[i, 0] = 1.0
                    Q[i, 1:1+n] = xi
                    idx = 1 + n
                    for j in range(n):
                        for k in range(j, n):
                            Q[i, idx] = xi[j] * xi[k]
                            idx += 1
                
                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                    if rank < Q.shape[1]:
                        # Rank deficient, might be unstable, but we try anyway with regularization
                        pass
                except np.linalg.LinAlgError:
                    coeffs = None
                
                if coeffs is not None:
                    c = coeffs[0]
                    b = coeffs[1:1+n]
                    
                    # Reconstruct A
                    A = np.zeros((n, n))
                    idx = 1 + n
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1
                    
                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                    # Add small regularization to handle singular matrices
                    A_reg = A + 1e-8 * np.eye(n)
                    try:
                        x_pred = -0.5 * np.linalg.solve(A_reg, b)
                    except np.linalg.LinAlgError:
                        x_pred = None
                    
                    # Check if x_pred is within bounds
                    if x_pred is not None and np.all(x_pred >= -5) and np.all(x_pred <= 5):
                        # Evaluate the predicted point
                        fx_pred = func(x_pred)
                        call_count += 1
                        history.append((x_pred, fx_pred))
                        
                        if fx_pred < f_best:
                            f_best = fx_pred
                            x_best = x_pred.copy()
                            # Increase sigma slightly if we made a big jump? Or keep it?
                            # Since we used the model, we can be a bit more aggressive
                            sigma = min(sigma * 1.1, max_sigma)
                        else:
                            # Model failed to improve, shrink sigma
                            sigma = max(sigma * 0.7, min_sigma)
                        continue
            
            # Fallback: Random perturbation
            # Generate a random step
            step = sigma * rng.standard_normal(n)
            x_new = x_best + step
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            call_count += 1
            history.append((x_new, fx_new))
            
            if fx_new < f_best:
                f_best = fx_new
                x_best = x_new.copy()
                sigma = min(sigma * 1.2, max_sigma)
            else:
                sigma = max(sigma * 0.8, min_sigma)
            
            # Optional: Prune history if it gets too large? 
            # For n=5 and budget 500, keeping all is fine memory-wise.
            # But for locality, we already select closest points.

        return x_best, f_best
# EVOLVE-BLOCK-END
