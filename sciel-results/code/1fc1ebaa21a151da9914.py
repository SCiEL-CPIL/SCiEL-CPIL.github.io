"""Model-and-Optimize optimizer with error repair for BBOB-style f6."""
import numpy as np
import time

class Optimizer:
    """Model-and-Optimize optimizer with error repair for BBOB-style f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Storage for all evaluated points and their function values
        X = []  # list of (dim,) arrays
        Y = []  # list of floats
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def evaluate_point(x):
            nonlocal best_x, best_f, calls_made
            # Error repair: clip to valid box
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            # Handle non-finite values (error repair)
            if not np.isfinite(f):
                f = 1e6
            X.append(x.copy())
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Initial sampling to build a diverse starting set
        # Use a mix of center, random, and Latin hypercube-like points
        n_init = min(20, budget)
        
        # Center point
        evaluate_point(np.zeros(dim))
        
        # Random points
        for _ in range(n_init - 1):
            x = rng.uniform(-5.0, 5.0, size=dim)
            evaluate_point(x)
        
        # Phase 2: Model-and-Optimize loop
        # Build a local quadratic surrogate model and optimize it
        # For dim=5, a full quadratic has 1 + 5 + 15 = 21 parameters
        # We need at least 21 points to fit it, but we can use least squares with regularization
        
        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Build design matrix for quadratic model
            # f(x) ≈ c0 + sum(ci*xi) + sum(cij*xi*xj)
            # We'll use a simpler approach: fit a quadratic using numpy's polyfit-like approach
            # But for multivariate, we'll construct the feature matrix manually
            
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            n_points = len(X_arr)
            
            if n_points < dim + 1:
                # Not enough points, sample randomly
                x_new = rng.uniform(-5.0, 5.0, size=dim)
                evaluate_point(x_new)
                continue
            
            # Construct quadratic feature matrix
            # Features: 1, x1, ..., x5, x1^2, ..., x5^2, x1*x2, ..., x4*x5
            # Total: 1 + 5 + 5 + 10 = 21 features
            
            def make_features(x):
                feats = [1.0]
                feats.extend(x)  # linear terms
                feats.extend(x**2)  # quadratic diagonal
                for i in range(dim):
                    for j in range(i+1, dim):
                        feats.append(x[i] * x[j])  # cross terms
                return feats
            
            # Build design matrix
            Phi = np.array([make_features(x) for x in X_arr])
            n_features = Phi.shape[1]
            
            # Fit quadratic model with ridge regression for stability
            # Minimize ||Phi @ theta - Y||^2 + lambda * ||theta||^2
            lambda_reg = 1e-3
            A = Phi.T @ Phi + lambda_reg * np.eye(n_features)
            b = Phi.T @ Y_arr
            theta = np.linalg.solve(A, b)
            
            # Now we need to find x that minimizes the surrogate:
            # g(x) = make_features(x) @ theta
            # This is a quadratic optimization problem. We can use scipy or a simple gradient descent.
            # For robustness, let's use a combination:
            # 1. Try to find the critical point analytically if possible
            # 2. Otherwise, use a few steps of gradient descent from multiple starts
            
            def surrogate(x):
                return make_features(x) @ theta
            
            def surrogate_grad(x):
                # Gradient of the quadratic surrogate
                # g(x) = c0 + sum(ci*xi) + sum(cii*xi^2) + sum(cij*xi*xj)
                # dg/dxi = ci + 2*cii*xi + sum_{j!=i} cij*xj
                grad = np.zeros(dim)
                # Linear terms: indices 1 to dim
                for i in range(dim):
                    grad[i] = theta[1 + i]
                # Quadratic diagonal terms: indices dim+1 to 2*dim
                for i in range(dim):
                    grad[i] += 2.0 * theta[dim + 1 + i] * x[i]
                # Cross terms: indices 2*dim+1 onward
                cross_idx = 2 * dim + 1
                for i in range(dim):
                    for j in range(i+1, dim):
                        # theta[cross_idx] corresponds to xi*xj
                        grad[i] += theta[cross_idx] * x[j]
                        grad[j] += theta[cross_idx] * x[i]
                        cross_idx += 1
                return grad
            
            # Find the minimum of the surrogate using multiple local optimizations
            # Start from the current best point and a few random points
            candidates = [best_x.copy()]
            for _ in range(3):
                candidates.append(rng.uniform(-5.0, 5.0, size=dim))
            
            best_candidate = None
            best_surrogate_val = np.inf
            
            for x0 in candidates:
                x0 = np.clip(x0, -5.0, 5.0)
                # Simple gradient descent with adaptive step size
                x = x0.copy()
                step_size = 0.5
                for _ in range(50):
                    g = surrogate_grad(x)
                    norm_g = np.linalg.norm(g)
                    if norm_g < 1e-8:
                        break
                    x_new = x - step_size * g
                    x_new = np.clip(x_new, -5.0, 5.0)
                    # Check if we improved
                    if surrogate(x_new) < surrogate(x) - 1e-10:
                        x = x_new
                        step_size = min(step_size * 1.2, 2.0)
                    else:
                        step_size *= 0.5
                        if step_size < 1e-6:
                            break
                
                val = surrogate(x)
                if val < best_surrogate_val:
                    best_surrogate_val = val
                    best_candidate = x.copy()
            
            # Evaluate the best candidate
            if best_candidate is not None:
                evaluate_point(best_candidate)
            else:
                # Fallback: random sample
                evaluate_point(rng.uniform(-5.0, 5.0, size=dim))
        
        return best_x, best_f
