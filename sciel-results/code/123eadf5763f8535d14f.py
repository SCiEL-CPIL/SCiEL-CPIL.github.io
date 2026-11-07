"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic fitting optimizer for Sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        # Phase 1: Quadratic fitting
        # We need d+1 points to fit a quadratic in d dimensions.
        # We use a local design around the initial point.
        # To be robust against the hidden shift/rotation, we use a local Taylor expansion.
        # Let x0 be the current point. We evaluate f at x0 + h * e_i for i=1..d.
        # This gives us the gradient and diagonal Hessian.
        # However, Sphere is rotated, so the Hessian is not diagonal in the standard basis.
        # A better approach for a general quadratic:
        # f(x) = x^T A x + b^T x + c
        # We need 1 + d + d(d+1)/2 points to determine A, b, c.
        # For d=5, that is 1 + 5 + 15 = 21 points.
        # This is well within the budget of 500.
        
        d = self.dim
        x0 = x.copy()
        f0 = fx
        
        # Design points for quadratic fit
        # We use a local grid.
        # h should be small enough to be in the quadratic regime (always for Sphere) 
        # but large enough to avoid numerical issues.
        h = 0.5
        
        # Points:
        # 1. x0
        # 2. x0 + h * e_i (d points)
        # 3. x0 + h * (e_i + e_j) / 2 (d(d-1)/2 points) -- wait, this doesn't span the cross terms properly for a general quadratic fit with linear algebra.
        
        # Let's use a standard design for quadratic regression.
        # We want to solve for A, b, c in f(x) = x^T A x + b^T x + c.
        # Actually, it's easier to fit f(x) = (x-x0)^T A (x-x0) + g^T (x-x0) + f0.
        # Let u = x - x0.
        # f(x0 + u) = u^T A u + g^T u + f0.
        # We need to determine A (symmetric dxd) and g (dx1).
        # Number of parameters: d(d+1)/2 + d.
        # For d=5: 15 + 5 = 20 parameters.
        # We need at least 20 points (plus x0 which is f0).
        
        # Let's generate 20 points in the neighborhood of x0.
        # We can use a random design or a structured one.
        # Structured: 
        # u_i = h * e_i (d points) -> gives diagonal of A and components of g.
        # u_ij = h/2 * (e_i + e_j) (d(d-1)/2 points) -> gives cross terms of A.
        # Total points: d + d(d-1)/2 = 5 + 10 = 15.
        # Wait, d(d-1)/2 for d=5 is 10. Total 15 points.
        # This gives 15 equations for 20 unknowns. Under-determined.
        # We need more points.
        
        # Let's just use 30 random points in a sphere of radius h around x0.
        # Or better, use a specific design that allows solving for A and g.
        
        # Alternative: Use the fact that Sphere is f(x) = ||R(x-s) + c||^2 ? No, f(x) = (x-x*)^T Q (x-x*) + f*.
        # Q is positive definite symmetric.
        
        # Let's collect 30 points.
        num_fit_points = 30
        if self.budget < num_fit_points + 1:
             # Fallback to simple ES if budget is tiny
             return self._fallback_es(func, x, fx, calls, deadline)
        
        # Generate design points
        # We want to estimate the gradient and Hessian at x0.
        # Central differences are more accurate but cost 2d.
        # Let's use a local quadratic model fit with random points.
        
        X = np.zeros((num_fit_points, d))
        F = np.zeros(num_fit_points)
        
        # First point is x0 itself? No, we already have f0.
        # Let's generate points in a sphere of radius 1.0 around x0.
        # We need to be careful not to go out of bounds [-5, 5].
        # x0 is in [-5, 5]. Radius 1.0 might go out if x0 is near boundary.
        # Clip points to [-5, 5].
        
        for i in range(num_fit_points):
            # Random direction
            dir = rng.standard_normal(d)
            dir /= np.linalg.norm(dir)
            # Random distance
            dist = rng.uniform(0.5, 1.5)
            u = dist * dir
            x_new = np.clip(x0 + u, -5, 5)
            X[i] = x_new - x0
            F[i] = func(x_new)
            calls += 1
            if time.time() > deadline:
                 break
        
        # Fit quadratic model: f(x0 + u) ≈ u^T A u + g^T u + f0
        # We have data (X, F).
        # Model: F ≈ X A X^T + g^T X + f0
        # Let y = F - f0.
        # y_i = X_i A X_i^T + g^T X_i
        # This is linear in the parameters of A and g.
        # Let's vectorize.
        # The term X_i A X_i^T can be written as sum_{j,k} A_{jk} X_{ij} X_{ik}.
        # So we build a design matrix M where each row corresponds to a point i.
        # The columns correspond to the parameters [A_{11}, A_{12}, ..., A_{dd}, g_1, ..., g_d].
        # Note A is symmetric, so A_{ij} = A_{ji}. We only include independent elements.
        
        d = self.dim
        # Number of independent A elements: d(d+1)/2
        # Number of g elements: d
        # Total params: p = d(d+1)/2 + d
        
        p = d * (d + 1) // 2 + d
        M = np.zeros((num_fit_points, p))
        
        # Fill M for A part
        idx = 0
        for i in range(num_fit_points):
            xi = X[i]
            for j in range(d):
                for k in range(j, d):
                    # Coefficient for A[j,k] in xi^T A xi is:
                    # If j==k: xi[j]^2
                    # If j!=k: 2 * xi[j] * xi[k] (because A[j,k] and A[k,j] both contribute, and they are equal)
                    # Wait, xi^T A xi = sum_j sum_k A[j,k] xi[j] xi[k]
                    # = sum_j A[j,j] xi[j]^2 + 2 sum_{j<k} A[j,k] xi[j] xi[k]
                    if j == k:
                        M[i, idx] = xi[j] * xi[j]
                    else:
                        M[i, idx] = 2 * xi[j] * xi[k]
                    idx += 1
            idx = 0 # Reset for g part? No, idx is for A part. Let's use separate indices.
        
        # Let's rebuild M more clearly.
        M = np.zeros((num_fit_points, p))
        
        # A part
        col = 0
        for j in range(d):
            for k in range(j, d):
                for i in range(num_fit_points):
                    xi = X[i]
                    if j == k:
                        M[i, col] = xi[j] * xi[j]
                    else:
                        M[i, col] = 2 * xi[j] * xi[k]
                col += 1
        
        # G part
        for j in range(d):
            for i in range(num_fit_points):
                M[i, col] = X[i, j]
            col += 1
            
        y = F - f0
        
        # Solve least squares: M theta = y
        # theta contains [A_11, A_12, ..., A_dd, g_1, ..., g_d]
        try:
            theta, residuals, rank, s = np.linalg.lstsq(M, y, rcond=None)
        except:
            return self._fallback_es(func, x, fx, calls, deadline)
            
        # Extract A and g
        A = np.zeros((d, d))
        col = 0
        for j in range(d):
            for k in range(j, d):
                val = theta[col]
                A[j, k] = val
                A[k, j] = val
                col += 1
        g = theta[col:col+d]
        
        # Find minimum of u^T A u + g^T u
        # Gradient: 2 A u + g = 0  => u* = -0.5 A^{-1} g
        # Hessian is 2A. If A is positive definite, this is a minimum.
        # Sphere function has positive definite Hessian.
        try:
            u_star = -0.5 * np.linalg.solve(A, g)
        except:
            # If A is singular, use pseudo-inverse or fallback
            u_star = -0.5 * np.linalg.pinv(A) @ g
            
        x_star = np.clip(x0 + u_star, -5, 5)
        f_star = func(x_star)
        calls += 1
        
        if f_star < fx:
            x, fx = x_star, f_star
        else:
            # If the fit was bad, keep the best found so far
            pass
            
        # Phase 2: Polish
        # Use a small step size local search around the best point found.
        remaining_calls = self.budget - calls
        if remaining_calls > 0:
            # Adaptive sigma based on distance to boundary or just small fixed
            sigma = 0.1
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                # Decrease sigma over time
                sigma *= 0.95
                if sigma < 1e-10:
                    sigma = 1e-10
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                # If we get stuck, maybe restart sigma?
                # For Sphere, simple local search is fine once close.
                
        return x, fx

    def _fallback_es(self, func, x, fx, calls, deadline):
        """Fallback (1+1)-ES if fitting fails or budget is too low."""
        d = self.dim
        rng = np.random.default_rng(1)
        sigma = 1.0
        remaining = self.budget - calls
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
        return x, fx
# EVOLVE-BLOCK-END
