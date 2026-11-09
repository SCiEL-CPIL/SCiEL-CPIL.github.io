"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with paraboloid fitting and coordinate descent polish for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_paraboloid(center, points, values):
            """Fit a quadratic model to points and find its minimum."""
            m = len(points)
            if m < n + 1:
                return None
            X = []
            y = []
            for p, v in zip(points, values):
                d = p - center
                row = [1.0]
                row.extend(d)
                for i in range(n):
                    for j in range(i, n):
                        row.append(d[i] * d[j])
                X.append(row)
                y.append(v)
            X = np.array(X)
            y = np.array(y)
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            c = coeffs[0]
            g = coeffs[1:n+1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        H[i, i] = 2.0 * coeffs[idx]
                    else:
                        H[i, j] = coeffs[idx]
                        H[j, i] = coeffs[idx]
                    idx += 1
            
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None
            
            model_at_center = c
            model_at_new = c + g @ dx + 0.5 * dx @ H @ dx
            if model_at_new >= model_at_center:
                return None
            
            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        # --- Phase 1: CMA-ES Global Search ---
        # Standard CMA-ES parameters
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initialize state
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        def get_C_sqrt(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-10 * np.eye(n))
                return L.T
            except:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        # Reserve budget for polish
        cmaes_budget_limit = total_budget - 60
        
        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break
            
            # Generate population
            z = np.random.standard_normal((lambda_, n))
            C_sqrt = get_C_sqrt(C)
            y = np.einsum('ij,jk->ik', z, C_sqrt.T)
            x_pop = mean + sigma * y
            
            # Evaluate population
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])
            
            if calls_made >= cmaes_budget_limit:
                break
                
            idx = np.argsort(f_pop)
            
            # Update mean
            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]
            
            # Update p_sigma
            try:
                C_inv_sqrt = np.linalg.inv(get_C_sqrt(C))
            except:
                C_inv_sqrt = np.eye(n)
                
            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)
            
            # Update p_c
            if np.linalg.norm(p_sigma) < (1.4 + 2/(n+1)) * np.sqrt((mueff+1)/(mueff+3)):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma
            
            # Update Covariance C
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
            
            # Update mean
            mean = x_w
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            if sigma < 1e-8:
                break

        # --- Phase 2: Paraboloid Fit & Local Polish ---
        # Use the best point found by CMA-ES
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Try a paraboloid fit to jump out of shallow local minima or refine basin center
            h = 0.5
            center = best_x.copy()
            points = [center]
            values = [best_f]
            for i in range(n):
                for sign in [1, -1]:
                    p = center.copy()
                    p[i] += sign * h
                    p = np.clip(p, -5, 5)
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    v = evaluate(p)
                    points.append(p)
                    values.append(v)
            
            jump = fit_paraboloid(center, points, values)
            if jump is not None:
                if calls_made < total_budget and time.time() < deadline:
                    fj = evaluate(jump)
                    if fj < best_f:
                        best_x = jump
                        best_f = fj

        # Coordinate Descent Polish
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            step_sizes = [0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005]
            
            for step in step_sizes:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                        # best_x updated in evaluate
                    
                    # Try negative direction
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True

                if not improved and step < 0.001:
                    break

        # Fallback
        if best_x is None:
            x = np.zeros(n)
            evaluate(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
