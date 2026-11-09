import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, x0, sigma0, budget_remaining, rng, deadline, d, calls_ref, best_ref):
        """Run CMA-ES from a given starting point. Returns (best_x, best_f, calls_used)."""
        lambda_pop = 4 + int(3 * np.log(d))
        mu = lambda_pop // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        x_m = x0.copy()
        C = np.eye(d)
        C_12 = np.eye(d)
        sigma = sigma0
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d**2))
        
        calls_used = 0
        best_f_local = float('inf')
        best_x_local = x0.copy()
        
        # Evaluate initial point
        f0 = float(func(x0))
        calls_used += 1
        calls_ref[0] += 1
        best_f_local = f0
        best_x_local = x0.copy()
        if f0 < best_ref[1]:
            best_ref[1] = f0
            best_ref[0] = x0.copy()
        
        # Stagnation tracking
        no_improve_count = 0
        stagnation_threshold = 10  # generations without improvement
        
        while calls_ref[0] + calls_used < budget_remaining and time.time() < deadline:
            # Check time
            if time.time() >= deadline:
                break
            
            # Sample offspring
            Z = rng.standard_normal((lambda_pop, d))
            Y = np.dot(Z, C_12.T)
            X = x_m + sigma * Y
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_used += lambda_pop
            calls_ref[0] += lambda_pop
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f_local:
                best_f_local = F[min_idx]
                best_x_local = X[min_idx].copy()
                if F[min_idx] < best_ref[1]:
                    best_ref[1] = F[min_idx]
                    best_ref[0] = X[min_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for stagnation
            if no_improve_count >= stagnation_threshold and calls_ref[0] < budget_remaining - lambda_pop:
                # Restart with perturbation
                break
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update mean
            y_w = np.dot(weights, X[:mu].T).T
            
            # Update sigma path
            y_sigma = np.dot(C_12.T, y_w / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * y_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used / lambda_pop))) < 1.4 * (1 + c_sigma) * chi_n else 0.0
            
            # Update covariance path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c)) * y_w
            
            # Update covariance matrix
            v = np.array(X[:mu] - x_m).T
            H = np.zeros((d, d))
            for i in range(mu):
                H += weights[i] * np.outer(v[:, i], v[:, i])
            
            H = H / (sigma**2)
            
            C = (1 - c_1 - c_mu) * C \
                + c_1 * ((1 + (1 - h_sigma) * c_c * (2 - c_c)) * np.outer(p_c, p_c) - C) \
                + c_mu * H
            
            # Ensure positive definiteness
            try:
                C_12 = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                # Fallback: reset to identity
                C = np.eye(d)
                C_12 = np.eye(d)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update mean
            x_m = x_m + sigma * y_w
            
            # Soft bound the mean to avoid extreme values
            x_m = np.clip(x_m, -10, 10)
        
        return best_x_local, best_f_local, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        rng = np.random.default_rng()
        
        # Track global best
        best_x = np.zeros(d)
        best_f = float('inf')
        calls_ref = [0]
        best_ref = [best_x, best_f]
        
        # Strategy: Multiple restarts with different initial points
        # Start with origin, then try other points if stagnation occurs
        
        # Initial exploration: evaluate several diverse starting points
        n_initial = min(5, budget // 10)
        initial_points = [np.zeros(d)]
        
        # Add some random points and structured points
        for i in range(n_initial - 1):
            if i % 2 == 0:
                # Random point in box
                p = rng.uniform(-3, 3, d)
            else:
                # Axis-aligned points
                p = np.zeros(d)
                p[i % d] = (-1)**(i // d) * 2.0
            initial_points.append(p)
        
        # Evaluate initial points
        for p in initial_points:
            if calls_ref[0] >= budget:
                break
            f = float(func(p))
            calls_ref[0] += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()
                best_ref[0] = p.copy()
                best_ref[1] = f
        
        # Run CMA-ES from the best initial point
        remaining_budget = budget - calls_ref[0]
        if remaining_budget > 10:
            x0 = best_x.copy()
            sigma0 = 1.0
            _, _, _ = self._cma_es_run(func, x0, sigma0, remaining_budget, rng, deadline, d, calls_ref, best_ref)
        
        # If still have budget and time, try another restart from a different point
        if calls_ref[0] < budget - 10 and time.time() < deadline - 10:
            # Try from a different random point
            x0_alt = rng.uniform(-4, 4, d)
            f_alt = float(func(x0_alt))
            calls_ref[0] += 1
            if f_alt < best_ref[1]:
                best_ref[0] = x0_alt.copy()
                best_ref[1] = f_alt
            
            remaining_budget = budget - calls_ref[0]
            if remaining_budget > 10:
                sigma0_alt = 0.5
                _, _, _ = self._cma_es_run(func, x0_alt, sigma0_alt, remaining_budget, rng, deadline, d, calls_ref, best_ref)
        
        return best_ref[0], best_ref[1]
