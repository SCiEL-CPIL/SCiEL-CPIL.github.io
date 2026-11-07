"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _gp_predict(self, X_train, y_train, X_pred, length_scale=1.0, noise=1e-6):
        """Simple GP prediction with RBF kernel."""
        n_train = X_train.shape[0]
        d = X_train.shape[1]
        
        # Compute kernel matrix
        K = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                dist_sq = np.sum((X_train[i] - X_train[j])**2)
                K[i, j] = np.exp(-0.5 * dist_sq / (length_scale**2))
        K += noise * np.eye(n_train)
        
        # Compute kernel vector between train and pred
        Ks = np.zeros((n_train, X_pred.shape[0]))
        for i in range(n_train):
            for j in range(X_pred.shape[0]):
                dist_sq = np.sum((X_train[i] - X_pred[j])**2)
                Ks[i, j] = np.exp(-0.5 * dist_sq / (length_scale**2))
        
        # Solve for mean and variance
        try:
            K_inv_y = np.linalg.solve(K, y_train)
            mean = Ks.T @ K_inv_y
            v = np.linalg.solve(K, Ks)
            var = 1.0 - np.sum(Ks * v, axis=0)
            var = np.maximum(var, 1e-10)
        except np.linalg.LinAlgError:
            mean = np.zeros(X_pred.shape[0])
            var = np.ones(X_pred.shape[0])
        
        return mean, var

    def _expected_improvement(self, mean, var, best_f, xi=0.01):
        """Compute expected improvement for minimization."""
        sd = np.sqrt(var)
        z = (best_f - mean - xi) / sd
        ei = (best_f - mean - xi) * norm.cdf(z) + sd * norm.pdf(z)
        return np.maximum(ei, 0)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        calls_used = 0
        X_obs = []
        y_obs = []
        best_x = None
        best_f = np.inf

        # Phase 1: Initial random sampling to build initial model
        n_initial = min(15, budget)
        for i in range(n_initial):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-4.5, 4.5, n)
            f = func(x)
            calls_used += 1
            X_obs.append(x)
            y_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Bayesian optimization loop
        # Use a subset of remaining budget for BO, save some for local polish
        bo_budget = min(300, budget - calls_used - 50)
        
        for iter_idx in range(bo_budget):
            if time.time() > deadline or calls_used >= budget - 50:
                break
            
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)
            
            # Generate candidate points
            n_candidates = 500
            X_cand = rng.uniform(-4.5, 4.5, (n_candidates, n))
            
            # GP prediction
            mean, var = self._gp_predict(X_arr, y_arr, X_cand, length_scale=1.5, noise=1e-5)
            
            # Compute EI
            ei = self._expected_improvement(mean, var, best_f, xi=0.005)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls_used += 1
            X_obs.append(x_new)
            y_obs.append(f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
                # If we found a good point, try local polish immediately
                if calls_used < budget - 20 and time.time() < deadline:
                    try:
                        res = minimize(
                            func,
                            x0=best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * n,
                            options={'maxiter': 50, 'maxfun': 1000, 'gtol': 1e-8}
                        )
                        calls_used += res.nfev
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass
        
        # Final local polish if we have budget
        if best_x is not None and calls_used < budget and time.time() < deadline:
            try:
                res2 = minimize(
                    func,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'maxfun': 100000, 'gtol': 1e-10}
                )
                calls_used += res2.nfev
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
