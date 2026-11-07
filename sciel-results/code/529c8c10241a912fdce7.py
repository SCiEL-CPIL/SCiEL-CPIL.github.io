import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
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
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Fit with minimal points (21)
        # For dim=5, we need 5 + 15 + 1 = 21 points.
        n_points = 2 * dim + 1
        
        # Generate points
        points = []
        # Center
        points.append(np.zeros(dim))
        
        # Axis-aligned points
        # Use a scale that is well within bounds but large enough for precision
        # The box is [-5, 5]. 2.5 is good.
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 2.5
            p2 = np.zeros(dim); p2[i] = -2.5
            points.append(p1)
            points.append(p2)
            
        # We have 1 + 2*dim points so far.
        # We need 21 total.
        # If dim=5, we have 11 points. We need 10 more.
        rng = np.random.default_rng(42)
        while len(points) < n_points:
            p = rng.uniform(-4, 4, dim)
            points.append(p)
            
        points = np.array(points)
        
        # Evaluate all points
        values = np.array([evaluate(p) for p in points])
        
        # Construct design matrix for quadratic fit
        # f(x) = sum_i sum_j Q_ij x_i x_j + sum_i q_i x_i + c
        def feature_vector(x):
            dim = len(x)
            quad_terms = []
            for i in range(dim):
                for j in range(i, dim):
                    quad_terms.append(x[i] * x[j])
            lin_terms = list(x)
            const = [1.0]
            return np.array(quad_terms + lin_terms + const)

        A = np.array([feature_vector(p) for p in points])
        b = values
        
        Q = None
        q = None
        x_opt = None
        
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            n_quad = dim * (dim + 1) // 2
            Q_flat = coeffs[:n_quad]
            q = coeffs[n_quad:n_quad + dim]
            
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    val = Q_flat[idx]
                    Q[i, j] = val
                    Q[j, i] = val
                    idx += 1
            
            # Solve Q x = -0.5 q
            # x_opt = -0.5 Q^{-1} q
            try:
                x_opt = np.linalg.solve(Q, -0.5 * q)
            except np.linalg.LinAlgError:
                x_opt = np.linalg.pinv(Q) @ (-0.5 * q)
            
            x_opt = np.clip(x_opt, -5, 5)
            f_opt = evaluate(x_opt)
            
            if f_opt < best_f:
                best_x = x_opt
                best_f = f_opt
                
        except Exception:
            x_opt = best_x

        # Phase 2: Polish with (1+1)-ES
        # Start from best_x
        x = best_x.copy()
        fx = best_f
        
        # Use Hessian to scale steps if possible
        H_inv = None
        if Q is not None:
            try:
                H = 2 * Q
                eigvals = np.linalg.eigvalsh(H)
                if np.all(eigvals > 1e-6):
                    H_inv = np.linalg.inv(H)
                    H_inv = (H_inv + H_inv.T) / 2
            except Exception:
                H_inv = None

        if H_inv is None:
            H_inv = np.eye(dim)

        try:
            L_inv = np.linalg.cholesky(H_inv)
        except np.linalg.LinAlgError:
            L_inv = np.eye(dim)

        # Adaptive sigma
        # Start small because we are likely very close to optimum
        sigma = 1e-3
        rng = np.random.default_rng(0)
        
        while calls_made < budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            step_vec = L_inv @ z
            norm = np.linalg.norm(step_vec)
            if norm < 1e-8:
                direction = np.ones(dim)
            else:
                direction = step_vec / norm
            
            y = np.clip(x + sigma * direction, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                
            if sigma < 1e-12:
                break

        return best_x, best_f
