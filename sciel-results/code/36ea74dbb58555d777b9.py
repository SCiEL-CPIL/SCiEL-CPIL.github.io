"""Optimizer for BBOB f6 (Attractive sector) using Surrogate-Assisted CMA-ES."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Surrogate-based optimizer for f6.
    1. Initial LHS sampling.
    2. Gaussian Process Surrogate with Expected Improvement acquisition.
    3. CMA-ES local polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget

        # Data storage for surrogate
        X_data = []
        Y_data = []

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            # Add to data for surrogate
            X_data.append(x_clipped)
            Y_data.append(f)
            return f

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling (Latin Hypercube)
        n_init = 50
        if n_init > budget:
            n_init = budget // 2
            
        # Generate LHS points
        # Simple LHS: divide each dimension into n_init bins, pick one random point per bin per dim
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            # Permute indices
            perm = rng.permutation(n_init)
            # Map to range [-5, 5]
            # Bin width is 10/n_init
            # Point in bin i is in [i*w, (i+1)*w]
            # We pick random offset within bin
            offsets = rng.random(n_init)
            lhs_points[perm, d] = -5.0 + (np.arange(n_init) + offsets) * (10.0 / n_init)
        
        for i in range(n_init):
            if time_left() < 5.0 or calls >= max_calls:
                break
            eval_func(lhs_points[i])

        if calls >= max_calls or time_left() < 10.0:
            return best_x, best_f

        # Phase 2: Surrogate-Based Optimization
        # We will use a subset of the budget for the surrogate loop
        # Reserve last 20% for CMA-ES polish
        cma_budget = max(50, int(budget * 0.2))
        surrogate_budget = max_calls - calls - cma_budget
        
        if surrogate_budget < 10:
            surrogate_budget = max_calls - calls
            cma_budget = 0

        # GP Parameters (Fixed for speed, tuned for 5D box [-5,5])
        # Length scale l=1.0, Variance sigma_f=1.0
        # We might need to scale the data if the function values are large, 
        # but EI is scale-invariant if we normalize Y.
        
        def compute_gp_params(X, Y):
            """Fit simple GP parameters (fixed hyperparams for stability/speed)"""
            # Standardize Y for better numerical stability in GP
            y_mean = np.mean(Y)
            y_std = np.std(Y)
            if y_std < 1e-8:
                y_std = 1.0
            Y_std = (Y - y_mean) / y_std
            
            n = len(X)
            d = X.shape[1]
            
            # RBF Kernel
            # l = 1.0, sigma_f = 1.0
            # Add small jitter for numerical stability
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_sq = np.sum((X[i] - X[j])**2)
                    K[i, j] = np.exp(-0.5 * dist_sq / (1.0**2))
            K += 1e-6 * np.eye(n)
            
            # Solve K @ alpha = Y_std
            try:
                alpha = np.linalg.solve(K, Y_std)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_std, rcond=None)[0]
                
            return K, alpha, y_mean, y_std

        def gp_predict(x_new, X, Y):
            """Predict mean and variance at x_new"""
            K, alpha, y_mean, y_std = compute_gp_params(X, Y)
            
            # Kernel vector k(x_new, X)
            k = np.zeros(len(X))
            for i in range(len(X)):
                dist_sq = np.sum((x_new - X[i])**2)
                k[i] = np.exp(-0.5 * dist_sq / (1.0**2))
                
            # Mean
            mu_std = np.dot(k, alpha)
            mu = mu_std * y_std + y_mean
            
            # Variance
            # var = 1 - k @ inv(K) @ k
            # We can solve K @ v = k to get v, then var = 1 - k @ v
            try:
                v = np.linalg.solve(K, k)
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(K, k, rcond=None)[0]
            
            var_std = 1.0 - np.dot(k, v)
            var = var_std * (y_std**2)
            
            # Clip variance to be positive
            if var < 0:
                var = 1e-8
                
            return mu, var

        def expected_improvement(x_new, current_best):
            """Compute EI for minimization"""
            X = np.array(X_data)
            Y = np.array(Y_data)
            
            mu, var = gp_predict(x_new, X, Y)
            sigma = np.sqrt(var)
            
            if sigma < 1e-8:
                return 0.0
            
            z = (current_best - mu) / sigma
            # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
            ei = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negate for minimization

        # Surrogate Loop
        # We will select points by optimizing EI
        # To speed up EI optimization, we use random restarts + L-BFGS-B
        
        for _ in range(surrogate_budget):
            if time_left() < 5.0 or calls >= max_calls:
                break
                
            # If we have very few points, just sample randomly
            if len(X_data) < 10:
                x_next = rng.uniform(-5.0, 5.0, dim)
            else:
                # Optimize EI
                # Use multiple random starting points for the EI optimizer
                best_ei_x = None
                best_ei_val = np.inf
                
                # 5 random restarts
                for _ in range(5):
                    x0 = rng.uniform(-5.0, 5.0, dim)
                    try:
                        res = minimize(
                            expected_improvement,
                            x0,
                            args=(best_f,),
                            method='L-BFGS-B',
                            bounds=[(-5.0, 5.0)] * dim,
                            options={'maxiter': 50, 'ftol': 1e-6}
                        )
                        if res.fun < best_ei_val:
                            best_ei_val = res.fun
                            best_ei_x = res.x
                    except Exception:
                        pass
                
                if best_ei_x is None:
                    x_next = rng.uniform(-5.0, 5.0, dim)
                else:
                    x_next = best_ei_x
            
            eval_func(x_next)

        if calls >= max_calls or time_left() < 10.0:
            return best_x, best_f

        # Phase 3: CMA-ES Polish
        remaining_budget = max_calls - calls
        if remaining_budget > 10:
            # Implement a lightweight CMA-ES
            def cmaes_polish(x0, sigma0, max_evals):
                nonlocal calls, best_f, best_x
                
                n = dim
                lambda_pop = 4 + int(3 * np.log(n))
                mu = lambda_pop // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                
                cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                cs = (mueff + 2) / (n + mueff + 5)
                c1 = 2 / ((n + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
                
                x_mean = x0.copy()
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = sigma0
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                
                def compute_sqrt_C(C):
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    return eigvecs @ np.diag(np.sqrt(eigvals))
                
                sqrt_C = compute_sqrt_C(C)
                
                n_evals = 0
                stagnant_generations = 0
                best_in_cma = best_f
                best_x_in_cma = best_x.copy()
                
                while n_evals < max_evals and time_left() > 5.0:
                    z = rng.standard_normal((lambda_pop, n))
                    offsprings = x_mean + sigma * (z @ sqrt_C.T)
                    
                    fitness = np.zeros(lambda_pop)
                    for i in range(lambda_pop):
                        if n_evals >= max_evals or time_left() < 5.0:
                            break
                        fitness[i] = eval_func(offsprings[i])
                        n_evals += 1
                    
                    indices = np.argsort(fitness)
                    x_sorted = offsprings[indices]
                    f_sorted = fitness[indices]
                    
                    if f_sorted[0] < best_f:
                        best_f = f_sorted[0]
                        best_x = x_sorted[0].copy()
                    
                    if f_sorted[0] < best_in_cma:
                        best_in_cma = f_sorted[0]
                        best_x_in_cma = x_sorted[0].copy()
                        stagnant_generations = 0
                    else:
                        stagnant_generations += 1
                    
                    x_mean_new = x_mean + sigma * sum(weights[i] * x_sorted[i] for i in range(mu))
                    
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * sqrt_C.T @ (x_mean_new - x_mean)
                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1) / lambda_pop)) < (c1 + 1) * chi_n + 0.3:
                        pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                    
                    diff = x_sorted[:mu] - x_mean
                    C = (1 + c1 * cmu - c1) * C + c1 * np.outer(pc, pc) + cmu * (diff * weights[:, None]).T @ diff
                    C = (C + C.T) / 2
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C += 1e-10 * np.eye(n)
                    
                    sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 1e10)
                    
                    x_mean = x_mean_new
                    sqrt_C = compute_sqrt_C(C)
                    
                    if np.linalg.norm(x_mean_new - x_mean) < 1e-12:
                        break
                    if stagnant_generations > 20:
                        break
                    if sigma < 1e-12:
                        break
                
                return best_x_in_cma, best_in_cma

            # Start CMA-ES from best point
            x_start = best_x.copy()
            # Initial sigma for polish should be small
            sigma_start = 0.5
            cmaes_polish(x_start, sigma_start, remaining_budget)

        return best_x, best_f
