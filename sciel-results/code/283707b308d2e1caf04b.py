import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from initial point x0 with initial step size sigma0."""
        n = self.dim
        # Population size: lambda = 10 for n=5
        lam = max(10, 4 + int(3 * n))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - (mueff + 2) / (n + 2)) / ((n + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        cc = (4 + n / mueff) / (n + 4 + 2 * mueff / n)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 2) / (n + 2)) - 1) + cs
        
        # State
        mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factor
        B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        
        # Chi_n for sigma update
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        global_best_x = mean.copy()
        global_best_f = func(global_best_x)
        eval_count = 1
        stagnation_counter = 0
        best_f_history = global_best_f
        
        while eval_count < budget and time.time() < deadline:
            # Generate population
            z = rng.standard_normal((lam, n))
            pop = mean[None, :] + sigma * (z @ B.T)
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fits = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    fits[i:] = np.inf
                    break
                fits[i] = func(pop[i])
                eval_count += 1
                if fits[i] < global_best_f:
                    global_best_f = fits[i]
                    global_best_x = pop[i].copy()
            
            # Sort
            idx = np.argsort(fits)
            pop_sorted = pop[idx]
            fits_sorted = fits[idx]
            
            # Update best tracking
            if fits_sorted[0] < best_f_history - 1e-10:
                best_f_history = fits_sorted[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Check for convergence
            if sigma < 1e-12 or stagnation_counter > 3 * lam:
                break
            
            # Compute y_mean in normalized space
            # Solve B * y = (x - mean) for each selected individual
            # y_i = B^{-1} (x_i - mean_old)
            # But we need to be careful: mean_old is the mean before update
            mean_old = mean.copy()
            
            # Update mean
            mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # y_mean in normalized coordinates
            y_mean = np.zeros(n)
            for i in range(mu):
                diff = pop_sorted[i] - mean_old
                y_i = np.linalg.solve(B, diff)
                y_mean += weights[i] * y_i
            
            # Update p_sigma
            norm_y_mean = np.linalg.norm(y_mean)
            h_sigma = norm_y_mean < (1.4 + 2/(n+1)) * chi_n
            if h_sigma:
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            h_c = np.linalg.norm(p_sigma) < (1.4 + 2/(n+1)) * (1 - cs) * chi_n
            if h_c:
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_mean
            
            # Update covariance matrix
            # Rank-one update
            C = (1 + cs * h_c * (c1 + cmu - c1) - c1) * C
            C += c1 * (np.outer(p_c, p_c) - C)
            # Rank-mu update
            Y = np.linalg.solve(B, (pop_sorted[:mu] - mean_old).T).T
            C += cmu * (np.sum(weights[:, None, None] * Y[:, :, None] * Y[:, None, :], axis=0) - C)
            
            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            try:
                B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-6 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    sigma = max(sigma, 0.5)
            
            mean = mean_new
        
        return global_best_x, global_best_f, eval_count

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan with Sobol points
        n_scan = 30
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        candidates = []
        for x in points:
            if time.time() > deadline - 5.0:
                break
            fx = func(x)
            candidates.append((fx, x.copy()))
        
        if not candidates:
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            return x0, fx0
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        best_x, best_f = candidates[0]
        
        # 2. Run CMA-ES from best point
        remaining_budget = self.budget - len(candidates)
        if remaining_budget > 20:
            # Initial sigma: based on spread of good candidates
            top_k = min(5, len(candidates))
            top_points = np.array([c[1] for c in candidates[:top_k]])
            spread = np.std(top_points, axis=0).mean()
            sigma0 = max(0.5, min(2.0, spread))
            
            x_best, f_best, used = self._cma_es(func, best_x, sigma0, remaining_budget, rng, deadline - 5.0)
            best_x, best_f = x_best, f_best
        
        # 3. Final polish: very small steps around best point
        remaining = self.budget - (len(candidates) + (self.budget - len(candidates) if remaining_budget > 20 else 0))
        if remaining > 5 and time.time() < deadline - 2.0:
            sigma_polish = 0.01
            for _ in range(min(remaining, 20)):
                if time.time() > deadline - 1.0:
                    break
                x_trial = best_x + sigma_polish * rng.standard_normal(self.dim)
                x_trial = np.clip(x_trial, -5, 5)
                fx = func(x_trial)
                if fx < best_f:
                    best_f = fx
                    best_x = x_trial.copy()
        
        return best_x, best_f
