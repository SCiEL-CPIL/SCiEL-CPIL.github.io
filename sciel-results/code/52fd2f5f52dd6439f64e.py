import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _matern52_kernel(self, X, Y, length_scale, variance):
        """Compute Matérn-5/2 kernel between points X and Y."""
        d = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)
        sqrt3d = np.sqrt(3) * d / length_scale
        K = variance * (1 + sqrt3d + (sqrt3d ** 2) / 3) * np.exp(-sqrt3d)
        return K

    def _expected_improvement(self, mu, sigma, best_f, xi=0.0):
        """Compute Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-8)
        imp = best_f - mu - xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[imp < 0] = 0.0
        return ei

    def _gp_predict(self, X_train, y_train, X_test, length_scale, variance, noise):
        """Gaussian Process prediction with Matérn-5/2 kernel."""
        K_train = self._matern52_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(len(X_train))
        K_test = self._matern52_kernel(X_test, X_train, length_scale, variance)
        
        try:
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            v = np.linalg.solve(L, K_test.T)
            mu = v.T @ alpha
            K_test_test = self._matern52_kernel(X_test, X_test, length_scale, variance)
            var = np.diag(K_test_test) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            return mu, sigma
        except np.linalg.LinAlgError:
            # Fallback: return prior
            mu = np.zeros(len(X_test))
            sigma = np.sqrt(variance) * np.ones(len(X_test))
            return mu, sigma

    def _bayesian_optimization_phase(self, func, best_x, best_f, n_iter, dim, deadline):
        """Perform Bayesian Optimization with Matérn-5/2 GP and EI acquisition."""
        calls_made = 0
        X = [best_x.copy()]
        y = [best_f]
        
        # Hyperparameters
        length_scale = 1.0
        variance = 1.0
        noise = 1e-4
        
        # Candidate points for acquisition
        rng = np.random.default_rng(123)
        
        for i in range(n_iter):
            if time.time() > deadline:
                break
            
            # Generate candidate points
            n_candidates = 200
            candidates = np.random.uniform(-5, 5, (n_candidates, dim))
            # Also include some points around current best
            local_candidates = best_x + rng.standard_normal((100, dim)) * 0.5
            local_candidates = np.clip(local_candidates, -5, 5)
            candidates = np.vstack([candidates, local_candidates])
            
            # GP prediction
            X_arr = np.array(X)
            y_arr = np.array(y)
            mu, sigma = self._gp_predict(X_arr, y_arr, candidates, length_scale, variance, noise)
            
            # EI acquisition
            ei = self._expected_improvement(mu, sigma, min(y_arr))
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = func(x_next)
            calls_made += 1
            
            X.append(x_next)
            y.append(f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        return best_x, best_f, calls_made

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

        # Phase 1: Initial exploration with deterministic points
        # Use a small set of well-spread points to get a good initial landscape
        n_init = 2 * dim + 1  # 11 points for dim=5
        
        points = []
        points.append(np.zeros(dim))
        
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 3.0
            p2 = np.zeros(dim); p2[i] = -3.0
            points.append(p1)
            points.append(p2)
        
        # Add a few random points for diversity
        rng_init = np.random.default_rng(42)
        while len(points) < n_init:
            p = rng_init.uniform(-4, 4, dim)
            points.append(p)
        
        for p in points:
            val = evaluate(p)
        
        # Phase 2: Bayesian Optimization for refined global search
        # Use about 30-50 evaluations for BO
        n_bo = min(40, budget - calls_made - 50)  # Leave some budget for local search
        if n_bo > 5:
            best_x, best_f, bo_calls = self._bayesian_optimization_phase(
                func, best_x, best_f, n_bo, dim, deadline
            )
            calls_made += bo_calls
        
        # Phase 3: Local polish with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        
        # Estimate Hessian-like scaling from recent evaluations if possible
        # For now, use identity with adaptive sigma
        sigma = 0.1
        rng = np.random.default_rng(0)
        
        # Try to estimate scale from BO results
        # If we have multiple evaluations, estimate typical step size
        if calls_made > 10:
            sigma = 0.05
        else:
            sigma = 0.2
        
        while calls_made < budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.3
            else:
                sigma *= 0.9
                
            if sigma < 1e-10:
                break
        
        return best_x, best_f
