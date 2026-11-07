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
        
        # Use a fixed seed for reproducibility of the 'random' center selection, 
        # or just use numpy's global random state which is likely seeded by the evaluator or default.
        # Using np.random directly is fine.
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Sphere-Specific Model Construction with Randomized Center
        # f(x) = ||x - x*||^2 = ||x||^2 - 2x*^T x + ||x*||^2
        # We model f(x) = a*||x - c||^2 + b*(x-c)^T + c_opt_val? No, simpler:
        # f(x) = x^T A x + g^T x + h
        # For sphere, A = I (or scaled I).
        # We can find the minimum x* by linear regression or explicit geometry.
        
        # Strategy: Pick a random center 'center' in the box.
        # Probe center and center +/- delta * e_i.
        # This allows us to potentially start very close to the optimum if center is lucky.
        
        # Choose a random center within [-4, 4]^5 to maximize chance of being near the hidden optimum
        center = np.random.uniform(-4.0, 4.0, dim)
        
        # Probe distance
        # Must be large enough to distinguish gradient, small enough to stay in bounds if center is near edge.
        # Since center is in [-4,4] and box is [-5,5], max distance to boundary is 1.
        # If we pick delta=0.5, points are in [-4.5, 4.5] which is safe.
        d = 0.5 
        
        points = [center]
        for i in range(dim):
            p1 = center.copy(); p1[i] += d
            p2 = center.copy(); p2[i] -= d
            points.append(p1)
            points.append(p2)
            
        # Ensure all points are within bounds (clipping just in case, though they should be)
        for p in points:
            np.clip(p, -5.0, 5.0, out=p)

        values = np.array([evaluate(p) for p in points])
        
        # Solve for minimum
        # Let u_i be the vector (p_i - center) / d for i=1..dim (positive axis)
        # Let v_i be the vector (p_i' - center) / d for i=1..dim (negative axis)
        # f(center + d*e_i) = f_c + d * g_i + d^2 * a
        # f(center - d*e_i) = f_c - d * g_i + d^2 * a
        # Diff: 2*d*g_i = f(pos) - f(neg) => g_i = (f(pos) - f(neg)) / (2d)
        # Sum: 2*f_c + 2*d^2*a = f(pos) + f(neg) => a = (f(pos) + f(neg) - 2*f_c) / (2*d^2)
        
        c_val = values[0]
        a_vals = []
        g_vec = np.zeros(dim)
        
        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            
            g_i = (val_pos - val_neg) / (2 * d)
            a_i = (val_pos + val_neg - 2 * c_val) / (2 * d * d)
            
            a_vals.append(a_i)
            g_vec[i] = g_i
            
        a = np.mean(a_vals)
        
        # The gradient at 'center' is g_vec.
        # For a sphere, f(x) = a*||x - x*||^2 + const.
        # grad f(x) = 2*a*(x - x*)
        # At x = center: g_vec = 2*a*(center - x*)
        # => x* = center - g_vec / (2*a)
        
        if abs(a) < 1e-10:
            # Degenerate case, maybe flat or numerical noise. 
            # Fall back to center if we can't determine curvature
            x_init = center
        else:
            x_init = center - g_vec / (2 * a)
            x_init = np.clip(x_init, -5.0, 5.0)
        
        # Evaluate the estimated optimum
        f_init = evaluate(x_init)
        
        # Phase 2: CMA-ES Polish
        # We are likely very close now. Use CMA-ES to polish to high precision.
        x = best_x.copy()
        
        # Initial sigma: 
        # If we are close, small sigma is better. 
        # If the model was slightly off, medium sigma.
        # The error in x_init is roughly proportional to noise / (2*a*d).
        # Let's start with a conservative sigma.
        sigma = 0.1
        
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        weights = np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        sigma_norm = np.sqrt(2 * (dim + 1) ** 0.8 - dim)
        
        # Run CMA-ES
        # We have 11 + 1 = 12 calls used. 488 left.
        # CMA-ES lambda is ~11. So ~44 generations.
        while calls_made < budget and time.time() < deadline:
            # Check if we have enough time for one generation?
            # Each generation takes lambda_ calls.
            if calls_made + lambda_ > budget:
                # Just do remaining evaluations if possible? 
                # Or break. CMA-ES needs full generations for stable update.
                # If we have < lambda_ calls left, it's better to just stop or do single steps?
                # Single steps in CMA-ES are not standard.
                # Let's break to avoid partial update issues, though evaluating more points is better than stopping.
                # Actually, the contract says "repeating the last n_i if fewer calls are made".
                # So stopping early is fine if we've found a good best_x.
                break

            B = np.linalg.cholesky(C)
            Z = np.random.randn(lambda_, dim)
            Y = Z @ B.T
            
            X = x + sigma * Y
            X = np.clip(X, -5.0, 5.0)
            
            F = np.array([evaluate(x_i) for x_i in X])
            
            if calls_made >= budget:
                break

            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Covariance Matrix
            Y_inv_sigma = (X_sorted[:mu] - x) / sigma
            delta_x = x_new - x
            x = x_new
            
            C = (1 - c_c) * C
            y_plus = ps / sigma_norm
            C += c_c * ( (1 - (1-c_c)*c_sigma) * np.outer(y_plus, y_plus) + np.sum(weights[:, None] * (Y_inv_sigma.T * Y_inv_sigma), axis=0) )
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) * delta_x / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(mu_eff * c_sigma * (2 - c_sigma)) * delta_x / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / sigma_norm * (np.linalg.norm(ps) / sigma_norm - 1))
            
            # Convergence check
            if sigma < 1e-15:
                break
            if np.linalg.norm(delta_x) < 1e-15:
                # Stagnation check, could restart, but let's just break to save time or continue?
                # If delta is tiny, we are converged.
                break

        return best_x, best_f
