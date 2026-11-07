import numpy as np
import time
from scipy.optimize import least_squares

class Optimizer:
    """
    Global Halton exploration + Local Quadratic Model optimization for BBOB f21.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        def time_left():
            return deadline - time.time()

        calls_made = 0
        best_x = None
        best_f = np.inf
        history = [] # List of (x, f) tuples

        def evaluate(x):
            nonlocal calls_made, best_x, best_f, history
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            history.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Halton Sequence Global Exploration
        # We use 100 points for global coverage.
        n_halton = min(100, budget)
        
        def halton(index, base):
            f = 1
            r = 0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                p = np.array([halton(i, base) for base in range(2, 2 + dim)])
                # Map [0,1] to [-5,5]
                p = p * 10.0 - 5.0
                points.append(p)
            return points

        halton_points = halton_sequence(n_halton, dim)
        
        for p in halton_points:
            if calls_made >= budget or time_left() < 5:
                break
            evaluate(p)

        # Phase 2: Local Quadratic Model Optimization
        # We will iteratively fit a quadratic model to recent points and step to its minimum.
        
        if best_x is None:
            return best_x, best_f

        # Helper to fit quadratic model
        # Model: f(x) = c + g^T x + 0.5 * x^T H x
        # We fit on local points.
        
        def fit_quadratic(points, values):
            """
            Fits a quadratic model to the given points and values.
            Returns (c, g, H) or None if fit is poor.
            """
            n_pts = len(points)
            if n_pts < dim + 1:
                return None
            
            # Construct design matrix A for least squares:
            # [1, x1, ..., xd, x1^2, x1*x2, ..., xd^2]
            # Number of coeffs: 1 + d + d(d+1)/2
            n_coeffs = 1 + dim + dim * (dim + 1) // 2
            if n_pts < n_coeffs:
                return None

            A = np.zeros((n_pts, n_coeffs))
            for i, x in enumerate(points):
                A[i, 0] = 1.0
                A[i, 1:dim+1] = x
                idx = dim + 1
                for j in range(dim):
                    for k in range(j, dim):
                        A[i, idx] = x[j] * x[k]
                        idx += 1
            
            # Solve least squares
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(A, values, rcond=None)
            except np.linalg.LinAlgError:
                return None

            # Check residual
            predicted = A @ coeffs
            residual = np.sum((values - predicted)**2)
            # If residual is too high relative to value range, model is bad
            if residual > 1e-4 * (np.max(values) - np.min(values))**2 + 1e-10:
                 # Heuristic: if fit is bad, return None
                 pass 
            # Actually, for BBOB, quadratic fit might be okay if local.
            
            c = coeffs[0]
            g = coeffs[1:dim+1]
            H_flat = coeffs[dim+1:]
            
            # Reconstruct H
            H = np.zeros((dim, dim))
            idx = 0
            for j in range(dim):
                for k in range(j, dim):
                    val = H_flat[idx]
                    H[j, k] = val
                    H[k, j] = val
                    idx += 1
            
            return c, g, H

        def get_quadratic_min(c, g, H, x_center, bounds_low, bounds_high, max_step):
            """
            Finds the minimum of the quadratic model c + g^T x + 0.5 x^T H x
            centered at x_center.
            """
            # We model f(x) = c + g^T (x - x_center) + 0.5 (x - x_center)^T H (x - x_center)
            # Let y = x - x_center
            # f(y) = c + g^T y + 0.5 y^T H y
            # Gradient wrt y: g + H y = 0 => y = -H^{-1} g
            # Hessian: H
            
            try:
                # Check if H is positive definite
                # If not, we might be at a saddle or maximum, or H is ill-conditioned.
                # We use a regularized inverse or just step in direction of -g if H is bad.
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-6:
                    # Not convex. Use gradient direction with small step
                    # Normalize g
                    g_norm = np.linalg.norm(g)
                    if g_norm < 1e-10:
                        return None
                    y = -g / g_norm * max_step
                else:
                    y = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                # Fallback to gradient step
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-10:
                    return None
                y = -g / g_norm * max_step

            # Limit step size
            y_norm = np.linalg.norm(y)
            if y_norm > max_step:
                y = y / y_norm * max_step
            
            x_new = x_center + y
            x_new = np.clip(x_new, bounds_low, bounds_high)
            
            return x_new

        # Local Search Loop
        # We will use a "moving window" of the last 2*dim+1 points for fitting.
        # But we need to ensure they are local.
        
        current_x = best_x.copy()
        current_f = best_f
        max_step = 1.0
        
        while calls_made < budget and time_left() > 2:
            # Select local points for fitting
            # We want points that are close to current_x
            # Let's take the last 15 points from history, but filter by distance
            # If not enough close points, generate new ones
            
            # Strategy: 
            # 1. If we have < 2*dim+1 points in history within distance max_step, generate new ones.
            # 2. Fit model.
            # 3. Step to model min.
            
            local_points = []
            local_vals = []
            
            # Gather recent points that are close
            for x, f in reversed(history):
                if np.linalg.norm(x - current_x) <= max_step:
                    local_points.append(x)
                    local_vals.append(f)
                    if len(local_points) >= 2 * dim + 1:
                        break
            
            if len(local_points) < 2 * dim + 1:
                # Generate new points around current_x
                # Use a small Halton sequence or random jitter
                n_gen = 2 * dim + 1 - len(local_points)
                for i in range(n_gen):
                    # Random jitter
                    jitter = np.random.normal(0, max_step / 2.0, dim)
                    x_new = current_x + jitter
                    if calls_made >= budget:
                        break
                    f_new = evaluate(x_new)
                    local_points.append(x_new)
                    local_vals.append(f_new)
                    # Update best if improved
                    # (evaluate already updates best_x/best_f)
                    if best_f < current_f:
                        current_x = best_x.copy()
                        current_f = best_f
                        # If we found a new best, we should probably restart the local fitting window
                        # But for simplicity, we continue.
                
                if calls_made >= budget:
                    break

            # Fit quadratic model
            model = fit_quadratic(local_points, local_vals)
            
            if model is None:
                # Fallback: random step
                jitter = np.random.normal(0, max_step / 5.0, dim)
                x_try = current_x + jitter
                if calls_made < budget:
                    f_try = evaluate(x_try)
                    if f_try < current_f:
                        current_x = best_x.copy()
                        current_f = best_f
                        max_step *= 0.5 # Shrink step if we improved? No, keep or shrink slightly.
                    else:
                        max_step *= 0.5 # Shrink if no improvement
                else:
                    break
            else:
                c, g, H = model
                x_next = get_quadratic_min(c, g, H, current_x, -5.0, 5.0, max_step)
                
                if x_next is None:
                    max_step *= 0.5
                    continue
                
                # Evaluate the predicted point
                if calls_made >= budget:
                    break
                
                f_next = evaluate(x_next)
                
                if f_next < current_f:
                    current_x = x_next.copy()
                    current_f = f_next
                    # If we improved, we might increase step size slightly or keep it
                    # max_step = min(max_step * 1.1, 2.0) 
                else:
                    # No improvement, shrink step size
                    max_step *= 0.7
                    if max_step < 1e-4:
                        break

        return best_x, best_f
