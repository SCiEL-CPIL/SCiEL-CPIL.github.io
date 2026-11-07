"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted global optimizer with local refinement for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        call_count = [0]
        eval_points = []  # list of (x, f) pairs
        eval_count = [0]

        def wrapped(x):
            nonlocal best_x, best_f
            eval_count[0] += 1
            if eval_count[0] > budget:
                return best_f if np.isfinite(best_f) else 0.0
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            eval_points.append((x.copy(), val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Initial sampling for surrogate ---
        # Sample a diverse set of points to build initial surrogate
        n_initial = min(30, budget // 5)
        
        # Include structured points
        init_points = []
        init_points.append(np.zeros(dim))
        # Corners
        for _ in range(min(8, n_initial - 1)):
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        # Some axis-aligned points
        for i in range(min(dim, n_initial - len(init_points))):
            pt = np.zeros(dim)
            pt[i] = rng.choice([-4.0, 4.0])
            init_points.append(pt)
        
        # Random fill
        while len(init_points) < n_initial:
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        for x0 in init_points:
            if eval_count[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            wrapped(x0)

        # --- Phase 2: Surrogate-guided exploration ---
        # Use a simple Gaussian Process surrogate with Expected Improvement
        # We'll implement a lightweight GP using RBF kernel
        
        def compute_gp_mean_var(X_new, X_train, y_train, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
            """Compute GP mean and variance at X_new given training data."""
            n_train = len(X_train)
            if n_train == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new)) * (sigma_f**2)
            
            # RBF kernel
            def rbf(X1, X2, ls):
                # Compute pairwise squared distances
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-sq_dists / (2 * ls**2))
            
            K = rbf(X_train, X_train, length_scale) + sigma_n * np.eye(n_train)
            K_star = rbf(X_new, X_train, length_scale)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mu = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = sigma_f**2 - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: use mean of training data
                mu = np.full(len(X_new), np.mean(y_train))
                var = np.full(len(X_new), np.var(y_train) + 1e-6)
            
            return mu, var

        def expected_improvement(mu, var, best_f_current, xi=0.01):
            """Compute Expected Improvement for minimization."""
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            z = (best_f_current - mu - xi) / sigma
            ei = (best_f_current - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        # Adaptive length scale based on data spread
        def estimate_length_scale(X_train):
            if len(X_train) < 2:
                return 1.0
            dists = []
            for i in range(min(len(X_train), 20)):
                for j in range(i+1, min(len(X_train), 20)):
                    dists.append(np.linalg.norm(X_train[i] - X_train[j]))
            if not dists:
                return 1.0
            return max(0.5, np.median(dists) * 0.5)

        # Main loop: alternate between surrogate-guided sampling and local refinement
        while eval_count[0] < budget and time.time() < deadline:
            if time.time() > deadline:
                break
            
            # Build training data
            if len(eval_points) >= 5:
                X_train = np.array([p[0] for p in eval_points])
                y_train = np.array([p[1] for p in eval_points])
                
                # Estimate length scale
                ls = estimate_length_scale(X_train)
                sigma_f = max(1e-6, np.std(y_train)) if len(y_train) > 1 else 1.0
                
                # Generate candidate points
                n_candidates = 50
                candidates = []
                
                # Random candidates
                for _ in range(n_candidates // 2):
                    candidates.append(rng.uniform(-4.8, 4.8, dim))
                
                # Perturbations around best
                if best_x is not None:
                    for scale in [0.1, 0.3, 0.5, 1.0]:
                        for _ in range(n_candidates // 4):
                            pert = best_x + rng.normal(0, scale, dim)
                            candidates.append(np.clip(pert, -4.8, 4.8))
                
                candidates = np.array(candidates)
                
                # Compute GP predictions
                mu, var = compute_gp_mean_var(candidates, X_train, y_train, ls, sigma_f)
                
                # Expected Improvement
                ei = expected_improvement(mu, var, best_f, xi=0.01)
                
                # Select top candidate
                best_idx = np.argmax(ei)
                x_next = candidates[best_idx]
            else:
                # Not enough data for GP, use random
                x_next = rng.uniform(-4.5, 4.5, dim)
            
            x_next = np.clip(x_next, -5.0, 5.0)
            wrapped(x_next)
            
            # Periodic local refinement: every 20 evaluations or so
            if eval_count[0] % 20 == 0 and best_x is not None and time.time() < deadline - 30:
                # Run a quick local optimization from best_x
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Final local refinement if time permits
        if best_x is not None and time.time() < deadline - 10 and eval_count[0] < budget:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            if eval_count[0] < budget:
                best_f = wrapped(best_x)
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
