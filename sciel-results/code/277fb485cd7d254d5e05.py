import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = np.zeros(n)
        best_fx = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Initial sampling to ensure we start in a reasonable location
        # Since f11 is unimodal, 0 is a good start, but let's check a few points
        rng = np.random.default_rng(42)
        for _ in range(5):
            pt = rng.uniform(-5, 5, n)
            fx = safe_call(pt)
            if fx is None: break
            if fx < best_fx:
                best_fx = fx
                best_x = pt.copy()
        
        # Also evaluate 0
        fx0 = safe_call(np.zeros(n))
        if fx0 is not None and fx0 < best_fx:
            best_fx = fx0
            best_x = np.zeros(n)

        # CMA-ES Implementation
        # Parameters for d=5
        lam = 9  # Population size
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 2.0  # Initial step size
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        while calls_made + lam <= budget and time.time() < deadline:
            # Generate population
            Y = rng.multivariate_normal(np.zeros(n), C, lam)
            X = x_mean + sigma * Y
            
            # Evaluate
            F = []
            for x in X:
                fx = safe_call(x)
                if fx is None:
                    F.append(np.inf)
                else:
                    F.append(fx)
            
            # Sort
            indices = np.argsort(F)
            X_sorted = X[indices]
            F_sorted = F[indices]
            Y_sorted = Y[indices]
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            y_mean = (x_mean_new - x_mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.inv(np.sqrt(C)) @ y_mean
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            delta_sig = (1 + 2 * max(0, (cc - 1) / (cc + 1))) * (np.linalg.norm(pc)**2 / cc - 1)
            C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + delta_sig * C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * Y_sorted[:mu, :, np.newaxis] * Y_sorted[:mu, :, np.newaxis].T, axis=0)
            
            # Ensure C is symmetric and positive definite (regularization if needed)
            C = (C + C.T) / 2
            # Simple eigenvalue check/clamp if it becomes singular
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            
            x_mean = x_mean_new
            gen += 1
            
            # Check convergence
            if sigma < 1e-12 or np.max(np.abs(y_mean)) < 1e-12:
                break

        # Final Polish with L-BFGS-B
        if calls_made < budget and time.time() < deadline:
            def polish_func(x):
                if time.time() > deadline or calls_made >= budget:
                    return np.inf
                fx = safe_call(x)
                if fx is None:
                    return np.inf
                return fx

            try:
                bounds = [(-5.0, 5.0)] * n
                res = minimize(polish_func, best_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'maxfun': 20})
            except Exception:
                pass

        return best_x, best_fx
