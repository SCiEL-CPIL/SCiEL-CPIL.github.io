import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 280.0  # Safety margin

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                # Ensure x is a 1D array of floats
                x = np.asarray(x, dtype=float).flatten()
                # Clip to box [-5, 5] to be safe, though func should do it
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        rng = np.random.default_rng(42)

        # CMA-ES implementation for Bent Cigar optimization
        # CMA-ES adaptively learns the covariance structure to align with
        # the narrow valley direction without gradient information
        
        def cma_es(start_x, initial_sigma, max_evals):
            """
            CMA-ES optimizer with adaptive covariance matrix.
            Handles the anisotropic curvature of Bent Cigar by learning
            the valley direction through population-based search.
            """
            n = d
            # CMA-ES parameters
            lambda_ = int(4 + 3 * np.log(n))  # population size
            mu = lambda_ // 2  # number of best individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initial state
            x_mean = start_x.copy()
            sigma = initial_sigma
            C = np.eye(n)  # covariance matrix
            pc = np.zeros(n)  # evolution path for covariance
            ps = np.zeros(n)  # evolution path for sigma
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            best_x_local = x_mean.copy()
            best_f_local = safe_eval(best_x_local)
            update_best(best_x_local, best_f_local)
            
            eval_count = 1
            
            while eval_count < max_evals and time_check() and self.calls < N:
                # Generate population
                pop = []
                fitnesses = []
                
                for i in range(lambda_):
                    if not time_check() or self.calls >= N:
                        break
                    # Sample from multivariate normal with covariance C
                    z = rng.standard_normal(n)
                    # Apply Cholesky decomposition of C
                    L = np.linalg.cholesky(C)
                    x_i = x_mean + sigma * (L @ z)
                    # Clip to bounds
                    x_i = np.clip(x_i, -5.0, 5.0)
                    
                    f_i = safe_eval(x_i)
                    eval_count += 1
                    
                    pop.append(x_i)
                    fitnesses.append(f_i)
                
                if len(pop) < 2:
                    break
                    
                # Sort by fitness
                sorted_idx = np.argsort(fitnesses)
                pop = [pop[i] for i in sorted_idx]
                fitnesses = [fitnesses[i] for i in sorted_idx]
                
                # Update best
                if fitnesses[0] < best_f_local:
                    best_f_local = fitnesses[0]
                    best_x_local = pop[0].copy()
                    update_best(best_x_local, best_f_local)
                
                # Compute weighted mean of best mu individuals
                x_new_mean = np.zeros(n)
                for i in range(mu):
                    x_new_mean += weights[i] * pop[i]
                
                # Update sigma path
                sigma_old = sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new_mean - x_mean)
                ps_norm = np.linalg.norm(ps) / chi_n
                
                # Update sigma
                if ps_norm < (1.4 + 2/(n+1)) * (1 - c_sigma):
                    sigma = sigma_old * np.exp(c_sigma / chi_n * (ps_norm - chi_n))
                
                # Update covariance path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                # Update covariance matrix
                delta_heavy = (1 - c_c) if (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(n+1)) * chi_n) else 0
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (pop[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * delta_heavy - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Add small regularization for numerical stability
                C += 1e-10 * np.eye(n)
                
                # Update mean
                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5.0, 5.0)
                
                # Check convergence
                if sigma < 1e-12:
                    break
                    # Also check if we're making progress
                if eval_count % (lambda_ * 5) == 0 and best_f_local < 1e-10:
                    break
                    
            return best_x_local, best_f_local

        # Phase 1: CMA-ES with moderate initial sigma to explore the landscape
        # Bent Cigar has a narrow valley, so we start with a reasonable sigma
        initial_sigma = 2.0
        max_cma_evals = min(N // 2, 200)
        
        x_cma, f_cma = cma_es(x0, initial_sigma, max_cma_evals)
        
        # Phase 2: If we have budget remaining, polish with L-BFGS-B
        if self.calls < N - 10 and time_check():
            try:
                res = minimize(
                    safe_eval, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 200, 'maxfun': N - self.calls - 5, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass
        
        # Phase 3: If still have budget, run another CMA-ES from the best point
        # with smaller sigma for fine-tuning
        if self.calls < N - 20 and time_check():
            fine_sigma = 0.5
            remaining = N - self.calls - 5
            if remaining > 20:
                x_fine, f_fine = cma_es(self.best_x, fine_sigma, remaining)
        
        return self.best_x, self.best_f
