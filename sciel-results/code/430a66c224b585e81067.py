import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        
        # Wrap func to track calls and enforce budget
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            # Clip to box to ensure validity, though func clips internally
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        bounds = [(-5.0, 5.0) for _ in range(n)]
        lb = np.array([-5.0] * n)
        ub = np.array([5.0] * n)
        
        # Initialize X and Y
        X = np.empty((0, n))
        Y = np.empty((0,))
        
        # 1. Initial Sobol Sampling for Global Coverage
        # Sobol provides low-discrepancy coverage, better than random for global search start
        n_init = 30
        # Use scipy's Sobol for quasi-random sampling
        sobol_gen = so.Sobol(d=n, scramble=True)
        # Skip first 2**d points for better uniformity in small dimensions? 
        # Actually, just generate n_init points.
        # Note: Sobol requires power of 2 for initial skip sometimes, but scramble handles it.
        # For small n_init, just use random or Sobol. Let's use Sobol.
        # scipy.optimize.Sobol is available in newer versions. 
        # If not, fallback to Halton or Random. 
        # Let's stick to a robust manual Halton or just random if Sobol import fails, 
        # but scipy is available. 
        # Actually, scipy.spatial.qhull or scipy.stats? 
        # Let's use a simple Halton sequence generator to avoid version issues with Sobol in scipy.optimize
        # Or just use random for simplicity if budget allows. 
        # With 500 budget, 30 random points is fine, but Sobol is better.
        
        # Let's implement a simple Halton sequence for initial points
        def halton_sequence(num_points, dim):
            points = []
            for i in range(1, num_points + 1):
                p = []
                for j in range(dim):
                    base = j + 2 # 2, 3, 5, 7...
                    f = 1.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        i_val = int(i_val / base)
                    p.append(f)
                points.append(p)
            return np.array(points)
        
        init_points = halton_sequence(n_init, n)
        # Map to bounds [-5, 5]
        init_points = lb + init_points * (ub - lb)
        
        for x in init_points:
            if not self._check_time() or self.calls >= self.budget:
                break
            y = safe_func(x)
            X = np.vstack([X, x])
            Y = np.append(Y, y)
            
        # 2. Bayesian Optimization Loop
        # Use Sparse GP with Matern 5/2 kernel (good for non-smooth Rastrigin components)
        # We will implement a simple sparse GP update using a subset of points (active set)
        # to keep computation O(M^3) where M << N.
        
        # Hyperparameters for Matern 5/2
        # length_scale, variance, noise
        # We'll fix reasonable defaults or do a very light optimization if time permits.
        # For speed, we'll fix length_scale=1.0, variance=1.0, noise=1e-4 initially.
        # If the function scale is unknown, we might need to normalize Y.
        
        # Normalize Y for GP stability
        y_mean = np.mean(Y) if len(Y) > 0 else 0
        y_std = np.std(Y) if len(Y) > 0 else 1
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std
        
        # Active set size for sparse GP
        max_active = 50
        
        # Kernel: Matern 5/2
        def matern52_kernel(X1, X2, length_scale):
            # X1: (M, D), X2: (N, D)
            # Returns (M, N)
            dist = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
            dist = np.maximum(dist, 1e-8) # avoid 0 division issues in exp? No, exp is fine.
            dist /= length_scale
            sqrt3_dist = np.sqrt(3) * dist
            K = (1 + sqrt3_dist + (3/5)*dist**2) * np.exp(-sqrt3_dist)
            return K
        
        # We will use a simple active set selection: points with highest posterior variance
        # or just the most recent points. 
        # For simplicity and robustness, let's use the most recent max_active points 
        # plus the current best point.
        
        # To improve, we can select active set based on distance to current best or variance.
        # Let's stick to recent + best for stability.
        
        # GP Prediction function
        def gp_predict(X_test, X_train, Y_train, length_scale=1.0, noise=1e-4):
            # X_train: (M, D), Y_train: (M,) normalized
            K_train = matern52_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
            K_test = matern52_kernel(X_test, X_train, length_scale)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K_train += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                except:
                    # Fallback: use mean
                    return np.full(len(X_test), np.mean(Y_train)), np.zeros(len(X_test))
            
            mu = K_test @ alpha
            v = np.linalg.solve(L, K_test.T)
            var = 1.0 - np.sum(v**2, axis=0) # Assuming unit variance prior
            var = np.maximum(var, 1e-8)
            
            return mu, var

        # Expected Improvement
        def expected_improvement(mu, var, best_y):
            # mu, var are normalized
            # best_y is normalized best
            improvement = best_y - mu
            sigma = np.sqrt(var)
            # Avoid division by zero
            sigma[sigma < 1e-8] = 1e-8
            z = improvement / sigma
            # Phi and phi
            # Use scipy.stats.norm
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            # Handle case where improvement < 0 (exploit vs explore)
            # Standard EI: max(0, improvement)
            # The formula above assumes normal dist.
            # If improvement < 0, cdf(z) is small, pdf(z) is small.
            # Actually, EI = E[max(0, f_best - f(x))]
            # = (f_best - mu) * Phi((f_best-mu)/sigma) + sigma * phi((f_best-mu)/sigma)
            # This formula is correct for the expectation of the positive part.
            return ei

        # Optimization of acquisition function
        # We will use L-BFGS-B to maximize EI
        # Since EI is non-differentiable at var=0, we add small jitter.
        
        # We need to select active set. 
        # Let's use a simple strategy: keep the 50 most influential points (highest variance at previous step) 
        # or just the most recent 50.
        # Given the budget (500), and GP cost O(M^3), M=50 is 125k ops, very fast.
        # Let's use the most recent 50 points.
        
        active_size = 50
        
        # Main BO Loop
        while self.calls < self.budget and self._check_time():
            # Select active set: most recent active_size points
            if len(X) > active_size:
                X_active = X[-active_size:]
                Y_active = Y_norm[-active_size:]
            else:
                X_active = X
                Y_active = Y_norm
            
            # Determine current best in normalized space
            best_y_norm = np.min(Y_active)
            
            # Define acquisition function to minimize (negative EI)
            def neg_ei(x):
                x = np.clip(x, lb, ub)
                # Predict at x
                mu, var = gp_predict(x.reshape(1, -1), X_active, Y_active)
                mu = mu[0]
                var = var[0]
                ei = expected_improvement(np.array([mu]), np.array([var]), best_y_norm)
                return -ei[0]
            
            # Find best acquisition point using multiple random restarts
            # L-BFGS-B from multiple starts to avoid local minima of EI
            best_acq_x = None
            best_acq_val = np.inf
            
            # Use 5 random restarts for the acquisition optimization
            # Start points: random + current best
            start_points = [self.best_x.copy()]
            for _ in range(4):
                start_points.append(rng.uniform(lb, ub))
            
            for x0 in start_points:
                try:
                    res = so.minimize(neg_ei, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20, 'ftol': 1e-4})
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_acq_x = res.x
                except Exception:
                    continue
            
            if best_acq_x is None:
                # Fallback to random
                best_acq_x = rng.uniform(lb, ub)
            
            # Evaluate the new point
            y_new = safe_func(best_acq_x)
            
            # Update X, Y, Y_norm
            X = np.vstack([X, best_acq_x])
            Y = np.append(Y, y_new)
            
            # Recalculate normalization? 
            # For stability, we can update mean/std incrementally or just recompute.
            # Recomputing is O(N), which is fine for N=500.
            y_mean = np.mean(Y)
            y_std = np.std(Y)
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y - y_mean) / y_std
            
            # If we have used a significant portion of the budget, 
            # we might want to switch to a local search or refine the GP.
            # But GP should handle it.
            
            # Optional: If variance is very low everywhere, stop?
            # The EI will naturally drop to 0.
            
            # Safety check: if we are stuck (no improvement in last 20 calls), 
            # maybe perturb the active set or restart GP?
            # For now, trust the GP.
            
            # To ensure we don't waste time on GP if it's slow, we check time.
            # The GP prediction is fast for M=50.
            
            # If we have very few calls left, do a final local polish
            if self.budget - self.calls < 50:
                # Break out of BO loop and do L-BFGS-B polish
                break
        
        # Phase 3: Final Polish
        if self.calls < self.budget and self._check_time() and self.best_x is not None:
            # Use L-BFGS-B from the best point found by BO
            x0 = self.best_x.copy()
            # Wrap func to stop if budget hit
            try:
                res_min = so.minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_min.fun < self.best_f:
                    self.best_f = res_min.fun
                    self.best_x = res_min.x
            except Exception:
                pass
        
        # Ensure return values are valid
        if self.best_x is None:
            self.best_x = np.zeros(n)
            self.best_f = safe_func(self.best_x)
            
        return self.best_x, self.best_f
