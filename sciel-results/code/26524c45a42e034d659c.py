import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 285.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        # Phase 1: Dense initial sampling for early AOCC
        # Use a mix of structured and random points
        n_initial = 20
        points = []
        # Center
        points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for val in [-2.0, 2.0]:
                p = np.zeros(dim)
                p[i] = val
                points.append(p)
        # Random points
        while len(points) < n_initial:
            p = rng.uniform(-5.0, 5.0, size=dim)
            points.append(p)

        for x in points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return best_x, float('inf')

        # Phase 2: CMA-ES style evolution strategy
        # We implement a lightweight CMA-ES variant
        x = best_x.copy()
        sigma = 1.0
        lambda_ = 4 * dim  # population size
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        # Step size control
        cs = 1 / (dim + 3)
        ds = 1 + 2 * max(0, (dim + 1) / (dim + 3) - 1)
        c1 = (2 / ((dim + 1.3)**2 + lambda_))
        cmu = min(0.5 - c1, 2 / ((dim + 2)**2 + lambda_))
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Memory for path updates
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # We need to manage the budget carefully.
        # Each generation uses lambda_ evaluations.
        # Total budget remaining: total_budget - calls_made
        
        generation = 0
        stall_counter = 0
        last_best_f = best_f

        while calls_made < total_budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if total_budget - calls_made < lambda_:
                # Use remaining budget for 1+1-ES steps
                break
            
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                y = L @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                offspring.append(x_new)

            # Evaluate offspring
            f_values = []
            for x_new in offspring:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                try:
                    f_val = wrapper(x_new)
                except TimeoutError:
                    f_val = float('inf')
                f_values.append(f_val)
            
            if len(f_values) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(f_values)
            mu = max(1, lambda_ // 2)  # number of parents for recombination
            # Weights for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)

            # Update mean
            x_old = x.copy()
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (offspring[indices[i]] - x)
            x = x + y_w
            x = np.clip(x, -5.0, 5.0)

            # Update sigma
            y_sigma = np.linalg.norm(y_w) / (sigma + 1e-16)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (y_w / (sigma + 1e-16))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update covariance
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (y_w / (sigma + 1e-16))
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (offspring[indices[i]] - x_old) / (sigma + 1e-16)
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * (mu_eff - 2) - c1) * C + c1 * rank_one + cmu * rank_mu
            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update best
            current_best_f = min(f_values)
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = offspring[indices[0]].copy()
                stall_counter = 0
            else:
                stall_counter += 1

            # If stalled for a while, try a restart or larger step
            if stall_counter > 5:
                # Increase sigma to escape local minimum
                sigma *= 1.5
                stall_counter = 0

            generation += 1

        # Phase 3: Final polish with Nelder-Mead if budget remains
        remaining = total_budget - calls_made
        if remaining > 20 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for final local optimization
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxfev': min(remaining, 50), 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
