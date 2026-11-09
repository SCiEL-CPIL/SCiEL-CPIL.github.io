import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan to find a good initial point
        # Use a moderate number of points to balance global exploration and leaving budget for CMA-ES
        n_scan = 50
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 10.0:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
            # Fallback if something went wrong with the scan
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            return x0, fx0

        # 2. CMA-ES Local Refinement
        # We use the best point from the scan as the initial mean.
        # Initial sigma is set to 1.0, which is a reasonable scale for the box [-5, 5]^5
        # assuming the minimizer is not extremely far from the initial point.
        
        # CMA-ES Parameters
        mu = self.dim // 2 + 1  # Number of parents
        lambda_ = 2 * mu        # Population size
        # We will use a simplified CMA-ES implementation suitable for small dimensions
        
        mean = best_x.copy()
        sigma = 1.0
        # Identity covariance matrix initially
        C = np.eye(self.dim)
        # Precompute weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Learning rates (standard recommendations)
        # cmu: weight of the successful parent
        cmu = 2 / ((self.dim + 1)**2 + 1)
        # cs: path length for covariance matrix adaptation
        cs = (mu + 0.5 + 2/(self.dim + 1)) / (self.dim + mu + 5)
        # c1: learning rate for rank-one update
        c1 = (2 - cmu + self.dim/15) / ((self.dim + 1.7)**2 + mu)
        # cc: learning rate for rank-one update of path
        cc = (4 + self.dim/mu) / (self.dim + 4 + 2*mu/self.dim)
        # damping
        # Hessian eigenvalue of the covariance matrix (approx 1 initially)
        damps = 1 + 2*max(0, np.sqrt((mu-2)/(self.dim+2)) - 1) + cs
        
        # Path lengths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Cholesky factor of C
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(self.dim)
        
        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        # Count evaluations
        eval_count = len(candidates)
        
        # Main CMA-ES Loop
        while eval_count < self.budget and time.time() < deadline - 5.0:
            
            # Generate population
            pop = np.zeros((lambda_, self.dim))
            for i in range(lambda_):
                # Standard normal
                z = rng.standard_normal(self.dim)
                # Transform using Cholesky
                # y = sigma * (mean + C^0.5 * z)
                y = mean + sigma * (B @ z)
                pop[i] = np.clip(y, -5, 5)
            
            # Evaluate population
            fit = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline - 2.0:
                    break
                fit[i] = func(pop[i])
                eval_count += 1
                if fit[i] < global_best_f:
                    global_best_f = fit[i]
                    global_best_x = pop[i].copy()
            
            # Sort by fitness
            idx = np.argsort(fit)
            fit_sorted = fit[idx]
            pop_sorted = pop[idx]
            
            # Update mean (recombination)
            mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update sigma (path length for step size adaptation)
            # Cholesky factor of C^{-1} (inverse)
            try:
                C_inv_chol = np.linalg.cholesky(np.linalg.inv(C))
            except:
                C_inv_chol = np.eye(self.dim)
            
            # Weighted sum of selected individuals (in normalized space)
            y_mean = np.zeros(self.dim)
            for i in range(mu):
                y_norm = np.linalg.solve(B, pop_sorted[i] - mean)
                y_mean += weights[i] * y_norm
            
            # Effective number of recombination weights
            mueff = 1 / np.sum(weights**2)
            
            # Update p_sigma
            chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            h_sigma = (np.linalg.norm(y_mean) / np.sqrt(1 - (1+iter_count)/(self.budget+1000)) < 1.4 + 2/(self.dim+1))
            if h_sigma:
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c (path length for covariance matrix)
            h_c = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1+iter_count)/(self.budget+1000)) < (1.4 + 2/(self.dim+1)) * (1 - cs))
            if h_c:
                p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(B, y_mean)
            
            # Update Covariance Matrix C
            # Rank-one update
            C = (1 + c1 * cs * h_c - c1) * C
            # Rank-mu update
            v = np.linalg.solve(B, pop_sorted[:mu] - mean)
            C += c1 * np.outer(p_c, p_c)
            C += cmu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
            
            # Ensure positive semi-definite by adding epsilon if needed
            # Recompute Cholesky
            try:
                B = np.linalg.cholesky(C + 1e-8 * np.eye(self.dim))
            except:
                # If Cholesky fails, reset C to identity and B to identity
                C = np.eye(self.dim)
                B = np.eye(self.dim)
            
            # Check for convergence/stagnation
            if sigma < 1e-6:
                break
                
            iter_count = getattr(self, 'iter_count', 0) + 1
            self.iter_count = iter_count

        return global_best_x, global_best_f
