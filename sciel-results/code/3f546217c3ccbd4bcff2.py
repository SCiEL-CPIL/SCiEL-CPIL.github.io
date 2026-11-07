"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer using quadratic surrogate for Attractive Sector."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        d = self.dim
        # Store evaluations: list of (x, f)
        evals = []
        
        def add_eval(x, f):
            evals.append((np.array(x, dtype=float), float(f)))

        # Phase 1: Initial exploration - sample points to fit quadratic model
        # Use a space-filling-ish initial set
        n_init = min(20, self.budget // 5)
        
        # Center point
        x0 = np.zeros(d)
        f0 = func(x0)
        add_eval(x0, f0)
        
        # Axis-aligned points
        for i in range(d):
            for sign in [1, -1]:
                xi = np.zeros(d)
                xi[i] = sign * 3.0
                fi = func(xi)
                add_eval(xi, fi)
        
        # Some random points
        n_rand = max(0, n_init - 1 - 2 * d)
        for _ in range(n_rand):
            xr = rng.uniform(-4, 4, d)
            fr = func(xr)
            add_eval(xr, fr)
        
        best_x = evals[0][0].copy()
        best_f = evals[0][1]
        for x, f in evals[1:]:
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Iterative model-based optimization
        # Fit quadratic model: f(x) ≈ c + b^T x + x^T A x
        # The Attractive Sector has known structure, so quadratic approximation should work well
        
        calls_made = len(evals)
        
        for iteration in range(100):  # Max iterations
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Build design matrix for quadratic fit
            # Features: 1, x_1, ..., x_d, x_1^2, ..., x_d^2, x_1*x_2, ..., x_{d-1}*x_d
            n_features = 1 + d + d + d * (d - 1) // 2
            
            def feature_vector(x):
                feats = [1.0]
                feats.extend(x)
                feats.extend(x ** 2)
                for i in range(d):
                    for j in range(i + 1, d):
                        feats.append(x[i] * x[j])
                return np.array(feats)
            
            X = np.array([feature_vector(x) for x, _ in evals])
            y = np.array([f for _, f in evals])
            
            # Fit least squares
            if len(evals) >= n_features:
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                except np.linalg.LinAlgError:
                    break
            else:
                # Not enough points, use ridge regression with small regularization
                try:
                    lam = 1e-6
                    coeffs = np.linalg.solve(X.T @ X + lam * np.eye(n_features), X.T @ y)
                except np.linalg.LinAlgError:
                    break
            
            # Define surrogate function
            def surrogate(x):
                fv = feature_vector(x)
                return float(coeffs @ fv)
            
            # Find minimum of surrogate using gradient-based optimization
            # Compute gradient of surrogate
            def surrogate_grad(x):
                # Manual gradient computation
                g = np.zeros(d)
                # Linear terms: coeffs[1:d+1]
                g += coeffs[1:d+1]
                # Quadratic terms: coeffs[d+1:2*d+1] for x_i^2
                for i in range(d):
                    g[i] += 2 * coeffs[d + 1 + i] * x[i]
                # Cross terms: coeffs[2*d+1:]
                cross_idx = 2 * d + 1
                for i in range(d):
                    for j in range(i + 1, d):
                        idx = cross_idx + (i * (d - 1) - i * (i + 1) // 2) + j
                        # Actually let's recompute index properly
                        # For i<j: index in cross terms
                        pass
                # Simpler: use numerical gradient
                eps = 1e-7
                g_num = np.zeros(d)
                for i in range(d):
                    xp = x.copy()
                    xp[i] += eps
                    xm = x.copy()
                    xm[i] -= eps
                    g_num[i] = (surrogate(xp) - surrogate(xm)) / (2 * eps)
                return g_num
            
            # Use L-BFGS-B to minimize surrogate
            from scipy.optimize import minimize
            
            # Try multiple starting points
            candidates = [best_x.copy()]
            # Add some random candidates
            for _ in range(5):
                candidates.append(rng.uniform(-4.5, 4.5, d))
            # Add current best neighborhood
            for _ in range(3):
                candidates.append(best_x + rng.normal(0, 0.5, d))
            
            best_surrogate_x = best_x.copy()
            best_surrogate_f = surrogate(best_x)
            
            for x_init in candidates:
                x_init_clipped = np.clip(x_init, -5, 5)
                try:
                    res = minimize(surrogate, x_init_clipped, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * d, options={'maxiter': 100, 'gtol': 1e-8})
                    if res.success or res.fun < best_surrogate_f:
                        if res.fun < best_surrogate_f:
                            best_surrogate_f = res.fun
                            best_surrogate_x = res.x
                except Exception:
                    continue
            
            best_surrogate_x = np.clip(best_surrogate_x, -5, 5)
            
            # Evaluate the true function at the surrogate minimum
            new_x = best_surrogate_x
            new_f = func(new_x)
            add_eval(new_x, new_f)
            calls_made += 1
            
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
            
            # If we're making progress, continue; otherwise, add diversity
            # Check if we've converged
            if iteration > 5:
                # Add a point in direction of most uncertainty (high variance region)
                # Simple approach: perturb the best point
                if calls_made < self.budget and time.time() < deadline:
                    perturbation = rng.normal(0, 0.3, d)
                    new_x2 = np.clip(best_x + perturbation, -5, 5)
                    new_f2 = func(new_x2)
                    add_eval(new_x2, new_f2)
                    calls_made += 1
                    if new_f2 < best_f:
                        best_f = new_f2
                        best_x = new_x2.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
