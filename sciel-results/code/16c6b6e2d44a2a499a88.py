"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic model fitting + high-precision polish for Sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Explicit deadline guard: 300s limit, use 295s for safety
        start_time = time.time()
        deadline = start_time + 295
        
        best_x = None
        best_fx = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Phase 1: Collect points to fit a quadratic model
        # Sphere function: f(x) = (R^T(x-c))^2 + f0 = x^T Q x + g^T x + c0
        # We need at least dim*(dim+1)/2 + dim + 1 = 16 points for dim=5
        # Use 25 points for stability
        
        # Initial points: origin + basis vectors + some random
        points = []
        fvals = []
        
        # Origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        points.append(x0)
        fvals.append(f0)
        
        # Basis vectors scaled to cover the domain
        scale = 4.0
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * scale
                fx = evaluate(x)
                points.append(x)
                fvals.append(fx)
        
        # Add some random points to ensure full rank and better conditioning
        while len(points) < 25:
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            points.append(x)
            fvals.append(fx)
        
        # Fit quadratic model
        # f(x) = x^T A x + b^T x + c
        # For each point x_i: sum_j A[i,j] * x_i[j] * x_i[i] ... wait, x^T A x = sum_i,j A[i,j] x_i x_j
        # Let's construct the design matrix properly.
        # y = X * theta + c
        # where theta contains A (symmetric) and b
        # Actually, simpler: 
        # f(x) = sum_i A_ii x_i^2 + sum_{i<j} 2 A_ij x_i x_j + sum_i b_i x_i + c
        # Let u be the vector of coefficients [A_11, ..., A_dd, A_12, ..., A_{d-1,d}, b_1, ..., b_d]
        # We can just use a general quadratic basis.
        
        n_pts = len(points)
        P = np.array(points)
        F = np.array(fvals)
        
        # Construct design matrix for quadratic terms
        # Columns: x_i^2 for all i, x_i x_j for i<j, x_i for all i
        # Number of cols: d + d*(d-1)/2 + d = 5 + 10 + 5 = 20
        # Plus intercept = 21 params. We have 25 points.
        
        cols = []
        # Square terms
        for i in range(dim):
            cols.append(P[:, i] ** 2)
        # Cross terms
        for i in range(dim):
            for j in range(i + 1, dim):
                cols.append(P[:, i] * P[:, j])
        # Linear terms
        for i in range(dim):
            cols.append(P[:, i])
        
        X = np.column_stack(cols)
        # Add intercept column
        X = np.column_stack([X, np.ones(n_pts)])
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, F, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = None
        
        if coeffs is not None:
            # Reconstruct Hessian H and gradient g
            # coeffs[0:d] -> H_ii / 2 ? 
            # f(x) = sum_i c_i x_i^2 + sum_{i<j} c_{ij} x_i x_j + sum_i c_{d+i} x_i + c_{2d}
            # H_ii = c_i
            # H_ij = c_{ij} / 2 (since x^T H x has 2 H_ij x_i x_j)
            # g_i = c_{d+i}
            
            H = np.zeros((dim, dim))
            g = np.zeros(dim)
            
            idx = 0
            for i in range(dim):
                H[i, i] = coeffs[idx]
                idx += 1
            for i in range(dim):
                for j in range(i + 1, dim):
                    val = coeffs[idx]
                    H[i, j] = val / 2.0
                    H[j, i] = val / 2.0
                    idx += 1
            for i in range(dim):
                g[i] = coeffs[idx]
                idx += 1
            
            # Find stationary point: H * x* + g = 0  => x* = -H^-1 g
            try:
                x_star = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                x_star = np.linalg.pinv(H) @ (-g)
            
            # Evaluate the candidate
            fx_star = evaluate(x_star)
            
            # If the model is good, we should be very close to the optimum now.
            # Check if we are still improving significantly.
            if best_fx - fx_star < 1e-6:
                # We are likely at the optimum. Start polishing.
                pass
            else:
                # Model might be poor. Do a few more random restarts or just proceed to polish from best_x
                pass

        # Phase 2: High-precision polish
        # Use a very fine-grained (1+1)-ES or coordinate descent
        # Since we have plenty of budget left (500 - ~26 = ~474 calls)
        # and the function is smooth and convex (Sphere), a simple local search will work.
        
        # Initialize from best_x found so far
        x = best_x.copy() if best_x is not None else np.zeros(dim)
        fx = best_fx
        
        sigma = 0.1
        no_improve_count = 0
        
        while calls < self.budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.05
                no_improve_count = 0
            else:
                sigma *= 0.95
                no_improve_count += 1
                
            # If sigma gets too small, we've converged
            if sigma < 1e-12:
                break
                
            # If we've had many failures, maybe we're stuck in a plateau or noise?
            # But Sphere is deterministic and smooth, so this shouldn't happen unless sigma is too large.
            # If no improve for a long time, reduce sigma more aggressively?
            if no_improve_count > 20:
                sigma *= 0.5
                no_improve_count = 0
                if sigma < 1e-15:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
