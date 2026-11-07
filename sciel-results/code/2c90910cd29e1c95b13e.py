"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global quadratic fit using a small number of strategically placed points."""

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
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Strategy:
        # The function is a rotated, shifted, and offset sphere (quadratic).
        # f(x) = (x-c)^T A (x-c) + f0
        # We can fit the quadratic parameters exactly using a small number of points.
        # The quadratic form has dim*(dim+1)/2 coefficients for the Hessian (symmetric),
        # dim coefficients for the linear term, and 1 for the constant.
        # Total parameters: dim*(dim+1)/2 + dim + 1.
        # For dim=5: 15 + 5 + 1 = 21 parameters.
        # We need at least 21 points to fit the system. To be robust, we use more.
        # A structured set of points:
        # 1. Origin: (0, ..., 0)
        # 2. Axis points: +/- h * e_i for i=1..dim (2*dim points)
        # 3. Pairwise points: h * (e_i + e_j) for i<j (dim*(dim-1)/2 points)
        # Total: 1 + 2*dim + dim*(dim-1)/2 = 1 + 10 + 10 = 21 points.
        # This is exactly the number of parameters.
        # However, the system might be ill-conditioned or the points might not span the space well if the shift is large.
        # Also, we don't know h. We can choose h=1.
        # But if the minimum is far away, the local quadratic approximation might be poor?
        # No, for a sphere, the quadratic approximation IS the function.
        # So fitting a global quadratic should work perfectly if we have enough points.
        
        # Let's generate the points:
        # p0 = 0
        # p_i = h * e_i, p_{dim+i} = -h * e_i
        # p_{2*dim + k} = h * (e_i + e_j) for i<j
        
        # We will evaluate these points.
        # Then we set up the linear system:
        # f(x) = x^T Q x + b^T x + c
        # where Q is symmetric.
        # We can write this as:
        # f(x) = sum_{i<=j} Q_ij x_i x_j + sum_i b_i x_i + c
        # Let's flatten the parameters.
        # For dim=5, there are 15 quadratic terms, 5 linear terms, 1 constant.
        
        # Let's use a least squares approach with a slightly larger set of points to be safe.
        # We can add random points or just use the structured set.
        # The structured set is deterministic and efficient.
        
        h = 1.0
        
        points = []
        points.append(np.zeros(dim))
        
        # Axis points
        for i in range(dim):
            p_plus = np.zeros(dim)
            p_plus[i] = h
            points.append(p_plus)
            
            p_minus = np.zeros(dim)
            p_minus[i] = -h
            points.append(p_minus)
        
        # Pairwise points
        for i in range(dim):
            for j in range(i+1, dim):
                p = np.zeros(dim)
                p[i] = h
                p[j] = h
                points.append(p)
        
        # Evaluate all these points
        f_values = []
        for p in points:
            if calls >= self.budget:
                break
            if time.time() >= deadline:
                break
            f_values.append(evaluate(p))
        
        # If we didn't get enough points, fallback to random search
        if len(f_values) < len(points):
            # Fallback: simple random search
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_fx
        
        # Now fit the quadratic
        # Design matrix A
        # Each row corresponds to a point x
        # Columns: x_i x_j (i<=j), x_i, 1
        n_points = len(points)
        n_params = dim*(dim+1)//2 + dim + 1
        
        A = np.zeros((n_points, n_params))
        b = np.array(f_values)
        
        for k, x in enumerate(points):
            col = 0
            # Quadratic terms
            for i in range(dim):
                for j in range(i, dim):
                    A[k, col] = x[i] * x[j]
                    col += 1
            # Linear terms
            for i in range(dim):
                A[k, col] = x[i]
                col += 1
            # Constant
            A[k, col] = 1.0
        
        # Solve A theta = b
        try:
            theta, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_fx
        
        # Reconstruct Q and b_vec
        # theta contains: Q_ij (i<=j), b_i, c
        Q = np.zeros((dim, dim))
        col = 0
        for i in range(dim):
            for j in range(i, dim):
                Q[i, j] = theta[col]
                Q[j, i] = theta[col]
                col += 1
        
        b_vec = theta[col:col+dim].copy()
        col += dim
        c = theta[col]
        
        # Find the minimum of f(x) = x^T Q x + b^T x + c
        # Gradient: 2 Q x + b = 0  => x* = -0.5 Q^{-1} b
        # If Q is singular or ill-conditioned, add ridge
        Q_reg = Q + 1e-6 * np.eye(dim)
        
        try:
            x_star = np.linalg.solve(Q_reg, -0.5 * b_vec)
        except np.linalg.LinAlgError:
            # Fallback
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_fx
        
        # Clip to box
        x_star = np.clip(x_star, -5, 5)
        
        # Evaluate the predicted minimum
        fx_star = evaluate(x_star)
        
        # If the prediction was bad, we might want to do a local search around the best point found so far
        # But for a sphere, the prediction should be very good.
        
        # Use remaining budget for a simple (1+1)-ES polish starting from x_star
        if calls < self.budget:
            x = x_star
            fx = fx_star
            sigma = 0.1
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                if calls >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
