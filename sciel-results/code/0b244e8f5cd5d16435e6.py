"""Rewritten optimizer: CMA-ES style adaptive search with quadratic surrogate acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and quadratic surrogate for fast local convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Initial population: mix of random and structured points
        rng = np.random.default_rng(42)
        
        # Start with a reasonable initial point
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # CMA-ES parameters
        lambda_ = max(4, dim + 3)  # population size
        mu = lambda_ // 2  # number of parents
        
        # Initial sigma
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Step size adaptation
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (sqrt_dim - 1) / (dim + 3)) if (sqrt_dim := np.sqrt(dim)) else 1.0
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0.0, (np.sqrt(dim) - 1) / (dim + 3))
        
        # Covariance adaptation
        c_c = (4 + mu / dim) / (dim + 4)
        c_cov = (2 + 2 * mu / dim) / (dim + dim / mu + 4)
        
        # Track best
        best_x = x0.copy()
        best_f = f0
        calls_used = 1
        
        # History for quadratic surrogate
        history_x = [x0.copy()]
        history_f = [f0]
        
        # Main loop
        while calls_used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate population
            pop_x = []
            pop_f = []
            
            for i in range(lambda_):
                # Check deadline
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform using C^0.5 (approximate via Cholesky)
                try:
                    L = np.linalg.cholesky(C)
                    step = sigma * L @ z
                except np.linalg.LinAlgError:
                    step = sigma * z
                
                x_new = x0 + step
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_used += 1
                
                pop_x.append(x_new)
                pop_f.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            if not pop_x:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            sorted_x = [pop_x[i] for i in indices]
            sorted_f = [pop_f[i] for i in indices]
            
            # Update mean (recombination)
            weights = np.ones(mu)
            weights[-1] = 2.0  # give extra weight to best
            weights /= weights.sum()
            
            x0_new = np.sum([w * x for w, x in zip(weights, sorted_x[:mu])], axis=0)
            x0_new = np.clip(x0_new, -5, 5)
            
            # Update sigma
            sigma_old = sigma
            diff = (x0_new - x0) / sigma_old if sigma_old > 1e-10 else np.zeros(dim)
            norm_diff = np.linalg.norm(diff)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * norm_diff * (1 if norm_diff > 1e-10 else 0)
            sigma *= np.exp(c_sigma * (norm_diff - np.sqrt(dim) * (1 + 0.5 * c_sigma)))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update covariance
            if norm_diff > 1e-10:
                p_c = (1 - c_c) * p_c + c_c * (x0_new - x0) / sigma_old
            
            rank_mu = np.zeros((dim, dim))
            for w, x in zip(weights, sorted_x[:mu]):
                v = (x - x0) / sigma_old if sigma_old > 1e-10 else np.zeros(dim)
                rank_mu += w * np.outer(v, v)
            
            rank_one = np.outer(p_c, p_c)
            C = (1 + c_cov * (c_c * (2 - c_c) - c_cov) - c_cov) * C \
                + c_cov * (rank_mu + (1 + c_c * (2 - c_c) - c_cov) * rank_one)
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x0 = x0_new
            
            # Quadratic surrogate acceleration
            # If we have enough points and are close to convergence, try fitting a paraboloid
            if len(history_x) >= dim + 1 and best_f < 1e3:
                # Add current best to history
                if not any(np.allclose(hx, best_x) for hx in history_x[-10:]):
                    history_x.append(best_x.copy())
                    history_f.append(best_f)
                
                # Keep only last 2*dim+1 points for fitting
                if len(history_x) > 2 * dim + 1:
                    history_x = history_x[-(2 * dim + 1):]
                    history_f = history_f[-(2 * dim + 1):]
                
                # Try to fit quadratic: f(x) ≈ f0 + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0)
                if len(history_x) >= dim + 1:
                    X = np.array(history_x)
                    Y = np.array(history_f)
                    
                    # Center around best point
                    Xc = X - best_x
                    Yc = Y - best_f
                    
                    # Design matrix for quadratic: [x_i, x_i^2]
                    # We'll fit f ≈ a^T x + b^T (x⊗x) + c
                    # Simplify: just do linear regression on gradient estimate
                    # For now, use a simple approach: fit linear model to estimate gradient
                    try:
                        # Add bias column
                        X_design = np.hstack([Xc, np.ones((len(Xc), 1))])
                        # Solve least squares
                        coeffs, _, _, _ = np.linalg.lstsq(X_design, Yc, rcond=None)
                        gradient = coeffs[:dim]
                        
                        # If gradient is small, we're near minimum
                        if np.linalg.norm(gradient) < 0.1:
                            # Take a small step in negative gradient direction
                            step_size = 0.1
                            x_trial = best_x - step_size * gradient
                            x_trial = np.clip(x_trial, -5, 5)
                            if calls_used < budget and time.time() < deadline:
                                f_trial = func(x_trial)
                                calls_used += 1
                                if f_trial < best_f:
                                    best_f = f_trial
                                    best_x = x_trial.copy()
                    except np.linalg.LinAlgError:
                        pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
