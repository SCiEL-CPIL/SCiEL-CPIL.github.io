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
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            # Clip to box to avoid any issues, though func does it too
            # pt = np.clip(pt, -5, 5) 
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Model Construction
        # We need dim*(dim+1)/2 + dim + 1 = 21 points for dim=5
        n_params = dim * (dim + 1) // 2 + dim + 1
        
        # Generate points
        # 1. Center
        # 2-11. +/- 2.0 * e_i (Scaled down to be safer if min is near center, 
        # but we need to span the space. 2.0 is good for conditioning if min is in [-4,4])
        # 12-21. Random points in [-3, 3]
        
        points = []
        points.append(np.zeros(dim))
        
        # Axis points
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 2.0
            p2 = np.zeros(dim); p2[i] = -2.0
            points.append(p1)
            points.append(p2)
            
        # Random points to fill up to n_params
        rng = np.random.default_rng(42)
        while len(points) < n_params:
            p = rng.uniform(-3.0, 3.0, dim)
            points.append(p)
            
        points = np.array(points)
        
        # Evaluate all points
        values = np.array([evaluate(p) for p in points])
        
        # Build feature matrix
        # Features: x_i*x_j (i<=j), x_i, 1
        A = np.zeros((n_params, n_params)) # Square system if we pick exactly n_params points
        for k, p in enumerate(points):
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    A[k, idx] = p[i] * p[j]
                    idx += 1
            for i in range(dim):
                A[k, idx] = p[i]
                idx += 1
            A[k, idx] = 1.0
        
        # Solve linear system
        try:
            coeffs = np.linalg.solve(A, values)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        
        # Extract parameters
        n_quad = dim * (dim + 1) // 2
        Q_flat = coeffs[:n_quad]
        q_vec = coeffs[n_quad:n_quad + dim]
        c_const = coeffs[n_quad + dim]
        
        # Reconstruct Q matrix (Hessian / 2)
        H = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                val = Q_flat[idx]
                H[i, j] = val
                H[j, i] = val
                idx += 1
                
        # H is the Hessian of the quadratic approximation.
        # f(x) = x^T H x + q^T x + c
        # grad f = 2 H x + q = 0 => x_opt = -0.5 H^{-1} q
        
        try:
            # Check if H is positive definite
            # If not, we might be in a saddle or non-convex region (unlikely for Sphere)
            # Use solve
            x_opt = np.linalg.solve(H, -0.5 * q_vec)
        except np.linalg.LinAlgError:
            x_opt = np.linalg.pinv(H) @ (-0.5 * q_vec)
            
        x_opt = np.clip(x_opt, -5, 5)
        
        # Evaluate the estimated optimum
        f_opt_est = evaluate(x_opt)
        
        # Phase 2: Local Polish
        # Use the Hessian to scale the steps for (1+1)-ES
        # This makes the search invariant to the shape of the quadratic
        # Step: x_new = x + sigma * H^{-1/2} * z
        
        # Compute H^{-1/2}
        try:
            # Eigen decomposition for H
            # H is symmetric
            eigvals, eigvecs = np.linalg.eigh(H)
            # Clip eigenvalues to avoid division by zero or negative issues
            eigvals = np.clip(eigvals, 1e-10, None)
            H_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except:
            H_inv_sqrt = np.eye(dim)
            
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma
        # The scale of the problem is determined by the eigenvalues of H.
        # A reasonable initial sigma is 1.0 in the whitened space.
        sigma = 1.0
        
        rng = np.random.default_rng(12345)
        
        # Polish loop
        while calls_made < budget and time.time() < deadline:
            # Generate random step in whitened space
            z = rng.standard_normal(dim)
            step = sigma * (H_inv_sqrt @ z)
            
            y = x + step
            y = np.clip(y, -5, 5)
            
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Success: increase sigma slightly
                sigma *= 1.1
            else:
                # Failure: decrease sigma
                sigma *= 0.9
                
            # Convergence check
            if sigma < 1e-12:
                break
            if fx < 1e-10: # If we are extremely close to 0 (assuming f_opt is small)
                 # But f_opt can be up to 100. 
                 # Better check: if step size is tiny
                 pass

        return best_x, best_f
