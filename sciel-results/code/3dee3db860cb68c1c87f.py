import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        # Wrap func to track calls and handle exceptions
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return best_f
            try:
                x = np.asarray(x).flatten()
                f = float(func(x))
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return float('inf')
        
        # Phase 1: Initial sampling to build the GP model
        # Use 20% of budget for initial exploration
        init_budget = max(10, int(total_budget * 0.2))
        
        # Latin Hypercube Sampling for initial points
        np.random.seed(42)  # For reproducibility in initialization
        X_init = np.random.uniform(-5, 5, size=(init_budget, dim))
        y_init = []
        
        for i in range(init_budget):
            if time.time() > deadline:
                break
            x = X_init[i]
            f = safe_func(x)
            y_init.append(f)
        
        y_init = np.array(y_init)
        
        # Phase 2: Bayesian Optimization with Gaussian Process
        # Use remaining budget for BO iterations
        remaining_budget = total_budget - calls_made
        
        if remaining_budget <= 0 or len(y_init) == 0:
            return best_x, best_f
        
        # Prepare data for GP
        X = X_init[:len(y_init)]
        y = y_init
        
        # GP parameters
        # We'll use a simple RBF kernel
        def kernel(x1, x2, length_scale=1.0, variance=1.0):
            """RBF kernel"""
            # Compute squared distances
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))
        
        # Simple GP implementation
        class SimpleGP:
            def __init__(self, X, y, length_scale=1.0, variance=1.0, noise=1e-6):
                self.X = X
                self.y = y
                self.length_scale = length_scale
                self.variance = variance
                self.noise = noise
                self.K = self._kernel_matrix(X, X)
                self.L = np.linalg.cholesky(self.K + self.noise * np.eye(len(X)))
                self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
                
            def _kernel_matrix(self, X1, X2):
                return kernel(X1, X2, self.length_scale, self.variance)
            
            def predict(self, X_new):
                """Predict mean and variance at new points"""
                K_star = self._kernel_matrix(X_new, self.X)
                mean = K_star @ self.alpha
                v = np.linalg.solve(self.L, K_star.T)
                var = self.variance - np.sum(v ** 2, axis=1)
                var = np.maximum(var, 1e-10)
                return mean, np.sqrt(var)
        
        # Initialize GP
        gp = SimpleGP(X, y, length_scale=1.0, variance=1.0)
        
        # Expected Improvement acquisition function
        def expected_improvement(x, current_best):
            """Calculate expected improvement for a point"""
            x = np.array([x])
            mu, sigma = gp.predict(x)
            mu = mu[0]
            sigma = sigma[0]
            
            if sigma < 1e-10:
                return 0.0
            
            improvement = current_best - mu
            z = improvement / sigma
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return ei
        
        # BO Loop
        bo_calls = 0
        max_bo_iterations = min(remaining_budget, 200)  # Limit BO iterations
        
        for iteration in range(max_bo_iterations):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            
            # Find the point that maximizes EI using local optimization
            # Use multiple random starting points for robustness
            best_ei = -np.inf
            best_x_candidate = np.zeros(dim)
            
            # Try several random starting points
            n_starts = 10
            for _ in range(n_starts):
                x0 = np.random.uniform(-5, 5, size=dim)
                
                def neg_ei(x):
                    return -expected_improvement(x, best_f)
                
                try:
                    res = minimize(neg_ei, x0, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < -best_ei:
                        best_ei = -res.fun
                        best_x_candidate = res.x
                except Exception:
                    pass
            
            # Clip to bounds
            best_x_candidate = np.clip(best_x_candidate, -5, 5)
            
            # Evaluate the candidate
            f_candidate = safe_func(best_x_candidate)
            bo_calls += 1
            
            # Update GP with new point
            if calls_made < total_budget:
                X = np.vstack([X, best_x_candidate])
                y = np.append(y, f_candidate)
                
                # Rebuild GP periodically to avoid numerical issues
                if len(X) % 10 == 0 or len(X) <= 20:
                    try:
                        gp = SimpleGP(X, y, length_scale=1.0, variance=1.0)
                    except Exception:
                        pass
        
        return best_x, best_f
