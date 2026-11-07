import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) > timeout

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Use lambda=10 as per inspiration
        N = 10
        rng = np.random.default_rng(42)
        mean = rng.uniform(-2, 2, d)
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix (identity)
        C = np.eye(d)
        C_chol = np.linalg.cholesky(C)
        
        # Weights
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff)))
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_N = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Evaluate initial mean
        f_mean = evaluate(mean)
        
        # CMA-ES Loop
        while calls_made + N < budget and not check_time():
            # Generate population
            pop = np.zeros((N, d))
            fvals = np.zeros(N)
            
            for i in range(N):
                z = rng.standard_normal(d)
                x = mean + sigma * (C_chol @ z)
                pop[i] = x
                fvals[i] = evaluate(x)
            
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            mean_new = np.dot(weights, pop[:mu])
            
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) / sigma * (mean_new - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean_new - mean)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_N * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update covariance
            delta_sigma = (np.linalg.norm(ps)**2 / chi_N**2 - 1)
            h_sigma = 1 if delta_sigma < (1 + 4/d) * chi_N**2 else 0
            
            y = (pop[:mu] - mean) / sigma
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * chi_N**2 * np.eye(d))
            
            # Rank-mu update
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])
            C += c_mu * rank_mu
            
            # Ensure positive definite
            C = (C + C.T) / 2.0
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(d)
                C_chol = np.linalg.cholesky(C)
                
            mean = mean_new

        # Polish Phase: Nelder-Mead
        if calls_made < budget and not check_time():
            # Use remaining budget for Nelder-Mead
            # Nelder-Mead typically uses ~ (d+1) * iterations function calls
            # We can set maxiter to be large and rely on time/budget checks in the wrapper
            # But scipy minimize doesn't easily accept a budget counter.
            # We can wrap the function to count calls and raise an exception or stop.
            
            def polish_func(x):
                nonlocal calls_made
                if calls_made >= budget or check_time():
                    raise ValueError("Budget or time exceeded")
                return evaluate(x)
            
            try:
                res = minimize(polish_func, best_x, method='Nelder-Mead', 
                               options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 100000})
                # Note: minimize might not respect the budget strictly if it doesn't check internally,
                # but our polish_func raises an exception.
                # If it raises, we keep the best_x found so far.
            except (ValueError, Exception):
                pass

        return best_x, best_f
