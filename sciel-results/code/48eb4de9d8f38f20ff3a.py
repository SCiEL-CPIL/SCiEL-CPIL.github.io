import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0
        
        # State
        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0
        
        # Helper to check time
        def time_exceeded():
            return time.time() - start_time > time_limit

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time_exceeded():
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Build initial quadratic model (2n+1 points)
        # Start at center
        x0 = np.zeros(n)
        f0 = evaluate(x0)
        if f0 is None:
            return best_x, best_f

        # Create simplex points: x0 + delta * e_i
        delta = 1.0
        points = [x0]
        values = [f0]
        for i in range(n):
            xi = x0.copy()
            xi[i] += delta
            fi = evaluate(xi)
            if fi is None:
                break
            points.append(xi)
            values.append(fi)
        
        if len(points) < n + 1:
            # Not enough budget for model, just return best
            return best_x, best_f

        # Estimate the minimizer z assuming f(x) = ||x - z||^2 + c
        # f(x_i) - f(x_0) = delta^2 - 2 * delta * z_i
        # z_i = (delta^2 - (f(x_i) - f(x_0))) / (2 * delta)
        z_est = np.zeros(n)
        for i in range(n):
            z_est[i] = (delta**2 - (values[i+1] - f0)) / (2 * delta)
            
        # Evaluate the estimated minimizer
        fm = evaluate(z_est)
        if fm is None:
            return best_x, best_f

        # Now we have a very good estimate of the minimizer.
        # We can use a local search or CMA-ES to refine it.
        # Given the quadratic nature, CMA-ES with a small sigma should converge fast.
        
        # Initialize CMA-ES state
        mean = z_est.copy()
        sigma = 0.2  # Start small since we are close
        C = np.eye(n)
        L = np.eye(n)
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Main loop
        while calls < budget:
            if time_exceeded():
                break
                
            # Generate population
            population = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                y = sigma * (L @ z)
                x = mean + y
                x = np.clip(x, -5, 5)
                population[i] = x
                
            f_values = np.zeros(lambda_)
            for i in range(lambda_):
                fx = evaluate(population[i])
                if fx is None:
                    break
                f_values[i] = fx
                if calls >= budget:
                    break
                    
            if calls >= budget:
                break
            
            # Sort
            idx = np.argsort(f_values)
            population = population[idx]
            f_values = f_values[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * population[:mu], axis=0)
            
            # Update step size and covariance
            try:
                L_old = np.linalg.cholesky(C)
                B_old = np.linalg.inv(L_old.T)
            except:
                B_old = np.eye(n)
                
            diff = (mean - new_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B_old @ diff)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (B_old @ diff)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            h_c = 1.0
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (population[i] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * np.eye(n)) + c_mu * C_update
            C = (C + C.T) / 2.0
            mean = new_mean
            
            try:
                L = np.linalg.cholesky(C)
            except:
                C += 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C)

        return best_x, best_f
