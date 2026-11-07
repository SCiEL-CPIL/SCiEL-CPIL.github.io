"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic Model-Based Optimizer for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        # 1. Initial Phase: Random Sampling to locate a good starting region
        # We need 2n+1 points for a full quadratic model, but we can start with fewer
        # and update. Let's do a small random search first to get a decent center.
        n_init = 20
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Quadratic Model Phase
        # Construct a local quadratic model: f(x) ~ c + g^T(x-x0) + 0.5(x-x0)^T H (x-x0)
        # We will maintain a set of points and fit the model.
        # Since we know it's likely a rotated sphere, H should be symmetric positive definite.
        
        # Strategy:
        # - Start with the best point found.
        # - Generate 2n new points around it to estimate the Hessian.
        # - Fit the quadratic model.
        # - Use the model to find the next step (trust region).
        # - Update the model with new points.
        
        # Initial points for Hessian estimation
        # Use a central difference-like approach or just random points in a sphere around best_x
        # To be robust, we'll collect points and use least squares to fit the quadratic.
        
        X = [best_x.copy()]
        F = [best_f]
        
        # Generate 2*dim points for initial Hessian estimate
        # We use a scaled identity direction approach for better conditioning initially
        scale = 0.5
        for i in range(dim):
            if time.time() > deadline: break
            dx = np.zeros(dim)
            dx[i] = scale
            x_plus = np.clip(best_x + dx, -5, 5)
            f_plus = func(x_plus)
            calls_made += 1
            X.append(x_plus)
            F.append(f_plus)
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            
            dx[i] = -scale
            x_minus = np.clip(best_x + dx, -5, 5)
            f_minus = func(x_minus)
            calls_made += 1
            X.append(x_minus)
            F.append(f_minus)
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
        
        # Fit the quadratic model
        def fit_quadratic(X, F, x0):
            """
            Fit f(x) = c + g^T(x-x0) + 0.5(x-x0)^T H (x-x0)
            Returns c, g, H
            """
            n = len(X)
            d = len(x0)
            # Design matrix A
            # Columns: 1, (x-x0)_1, ..., (x-x0)_d, (x-x0)_i*(x-x0)_j for i<=j
            # Total columns: 1 + d + d*(d+1)/2
            n_q = d * (d + 1) // 2
            n_cols = 1 + d + n_q
            
            A = np.zeros((n, n_cols))
            A[:, 0] = 1.0
            
            # Linear terms
            for i in range(d):
                A[:, 1 + i] = X[:, i] - x0[i]
            
            # Quadratic terms
            col_idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    A[:, col_idx] = (X[:, i] - x0[i]) * (X[:, j] - x0[j])
                    col_idx += 1
            
            # Solve least squares
            try:
                params, residuals, rank, sv = np.linalg.lstsq(A, np.array(F), rcond=None)
            except np.linalg.LinAlgError:
                # Fallback to identity if fitting fails
                return F[0], np.zeros(d), np.eye(d)
            
            c = params[0]
            g = params[1:1+d].copy()
            
            # Reconstruct H
            H = np.zeros((d, d))
            col_idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    val = params[col_idx]
                    H[i, j] += val
                    if i != j:
                        H[i, j] += val
                    col_idx += 1
            
            # Ensure symmetry
            H = (H + H.T) / 2.0
            
            # Regularize H to be positive definite
            # Add small identity if needed
            eigvals = np.linalg.eigvalsh(H)
            if np.min(eigvals) < 1e-6:
                H += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(d)
            
            return c, g, H

        # Main Loop
        # We will iteratively:
        # 1. Fit model around current best_x
        # 2. Find minimizer of model within a trust region
        # 3. Evaluate new point
        # 4. Update best
        # 5. Add new point to X, remove worst point (keep size manageable)
        
        # Keep X size limited to 50 points for efficiency
        max_X_size = 50
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Fit model around best_x
            c, g, H = fit_quadratic(np.array(X), np.array(F), best_x)
            
            # Find step delta minimizing model:
            # m(delta) = c + g^T delta + 0.5 delta^T H delta
            # Gradient: g + H delta = 0 => delta = -H^-1 g
            try:
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                delta = -g * 0.1 # Fallback
            
            # Trust Region: limit step size
            # Start with a reasonable radius, adapt based on success
            # Initial radius 1.0
            trust_radius = 1.0
            
            # If step is too large, shrink
            step_norm = np.linalg.norm(delta)
            if step_norm > trust_radius:
                delta = delta * (trust_radius / step_norm)
            
            # Check if step is too small (stagnation)
            if step_norm < 1e-6:
                # Perturb
                delta = rng.normal(0, 0.1, dim)
            
            # Propose new point
            x_new = np.clip(best_x + delta, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < best_f:
                # Success: expand trust region
                best_f = f_new
                best_x = x_new.copy()
                trust_radius *= 1.5
                # Add to X
                X.append(x_new)
                F.append(f_new)
                
                # If X is too large, remove the worst point (highest f)
                if len(X) > max_X_size:
                    idx_worst = np.argmax(F)
                    X.pop(idx_worst)
                    F.pop(idx_worst)
            else:
                # Failure: shrink trust region
                trust_radius *= 0.5
                # Don't add to X, or add with penalty? 
                # For quadratic fitting, adding bad points can distort the model.
                # Only add if it's close to best? No, keep model clean.
                # However, if we keep failing, we might be stuck.
                # If trust_radius becomes very small, we might need to restart or random jump.
                if trust_radius < 1e-4:
                    # Random jump to escape local minimum (though sphere has only one)
                    # Or just accept that we've converged
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
