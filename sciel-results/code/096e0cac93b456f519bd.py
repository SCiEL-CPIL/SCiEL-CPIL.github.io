"""Hybrid surrogate-assisted BO with multi-start adaptive (1+1)-ES for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Hybrid surrogate-assisted Bayesian Optimization with local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ========================================================================
        # Phase 1: Global Exploration using Sobol Sequence
        # ========================================================================
        n_init = 30
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n=n_init)
        init_points = sobol_points * 10.0 - 5.0

        init_results = []
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(p)
            init_results.append((p, f))

        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])

        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # ========================================================================
        # Phase 2: Surrogate-Assisted Bayesian Optimization
        # ========================================================================
        def compute_kr(x1, x2, theta):
            """Compute Kronecker product RBF kernel."""
            l, nu = theta
            d = np.sum((x1 - x2) ** 2)
            return nu ** 2 * np.exp(-0.5 * d / (l ** 2))

        def log_likelihood(theta, X, y):
            """Negative log marginal likelihood for GP."""
            try:
                l, nu = theta
                if l <= 0 or nu <= 0:
                    return 1e6
                n = len(y)
                K = np.array([[compute_kr(X[i], X[j], theta) for j in range(n)] for i in range(n)])
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    return -ll
                except:
                    return 1e6
            except:
                return 1e6

        def gp_predict(X_train, y_train, X_test, theta):
            """GP mean and variance prediction."""
            n = len(y_train)
            K = np.array([[compute_kr(X_train[i], X_train[j], theta) for j in range(n)] for i in range(n)])
            K += 1e-6 * np.eye(n)
            K_star = np.array([[compute_kr(x, X_train[j], theta) for j in range(n)] for x in X_test])
            try:
                alpha = np.linalg.solve(K, y_train)
                mean = K_star @ alpha
                v = np.linalg.solve(K, K_star.T)
                var = compute_kr(np.zeros(dim), np.zeros(dim), theta) - np.sum(K_star ** 2, axis=1)
                var = np.maximum(var, 1e-8)
                return mean, var
            except:
                mean = np.mean(y_train)
                var = np.var(y_train)
                return np.full(len(X_test), mean), np.full(len(X_test), var)

        def expected_improvement(mean, var, f_best, xi=0.01):
            """Expected improvement acquisition function."""
            var = np.maximum(var, 1e-8)
            z = (f_best - mean - xi) / np.sqrt(var)
            ei = (f_best - mean - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            return ei

        def fit_gp(X, y):
            """Fit GP hyperparameters using scipy minimize."""
            if len(X) < 2:
                return np.array([1.0, 1.0])
            
            # Normalize inputs for better conditioning
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = np.maximum(X_max - X_min, 1e-6)
            X_norm = (X - X_min) / X_range
            
            y_mean = np.mean(y)
            y_std = max(np.std(y), 1e-6)
            y_norm = (y - y_mean) / y_std
            
            def objective(log_theta):
                theta = np.exp(log_theta)
                return log_likelihood(theta, X_norm, y_norm)
            
            # Initial guess in log-space
            x0 = np.array([0.0, 0.0])  # l=1, nu=1
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
                theta = np.exp(res.x)
                # Clip to reasonable range
                theta[0] = np.clip(theta[0], 0.1, 5.0)  # length scale
                theta[1] = np.clip(theta[1], 0.1, 10.0)  # signal variance
                return theta
            except:
                return np.array([1.0, 1.0])

        def run_bayesian_optimization(x0, f0, max_calls, initial_theta=None):
            """Run BO loop starting from x0, f0."""
            nonlocal calls_made, best_x, best_f
            
            X_obs = [x0.copy()]
            y_obs = [f0]
            
            if initial_theta is not None:
                theta = initial_theta
            else:
                theta = np.array([1.0, 1.0])
            
            for step in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Refit GP if we have enough data
                if len(X_obs) >= 5 and step % 5 == 0:
                    X_arr = np.array(X_obs)
                    y_arr = np.array(y_obs)
                    theta = fit_gp(X_arr, y_arr)
                
                # Generate candidate points
                n_candidates = 50
                cand_points = []
                
                # Some random exploration
                n_rand = 10
                rand_pts = rng.uniform(-5, 5, size=(n_rand, dim))
                cand_points.extend(rand_pts)
                
                # Some points near existing data
                if len(X_obs) > 0:
                    for _ in range(10):
                        idx = rng.choice(len(X_obs))
                        jitter = rng.normal(0, 0.5, size=dim)
                        cand_points.append(np.clip(X_obs[idx] + jitter, -5, 5))
                
                # Some Sobol-like points
                sampler = qmc.Sobol(d=dim, scramble=True)
                sobol_pts = sampler.random(n=20) * 10 - 5
                cand_points.extend(sobol_pts)
                
                cand_points = np.array(cand_points)
                cand_points = np.clip(cand_points, -5, 5)
                
                # Evaluate acquisition function
                if len(X_obs) >= 2:
                    X_arr = np.array(X_obs)
                    y_arr = np.array(y_obs)
                    mean, var = gp_predict(X_arr, y_arr, cand_points, theta)
                    f_best_current = min(y_obs)
                    ei = expected_improvement(mean, var, f_best_current, xi=0.01)
                    best_idx = np.argmax(ei)
                else:
                    # Just pick random
                    best_idx = rng.choice(len(cand_points))
                
                x_new = cand_points[best_idx]
                f_new = evaluate(x_new)
                
                X_obs.append(x_new.copy())
                y_obs.append(f_new)
                
                # Early stopping if no improvement for a while
                if len(y_obs) > 20:
                    recent = y_obs[-20:]
                    if min(recent) >= min(y_obs) - 1e-6:
                        break
            
            return min(y_obs)

        # ========================================================================
        # Phase 3: Multi-Start Adaptive (1+1)-ES
        # ========================================================================
        def run_11es(x0, f0, max_calls):
            """Run adaptive (1+1)-ES with 1/5 success rule."""
            nonlocal calls_made, best_x, best_f
            
            x = x0.copy()
            f = f0
            sigma = 0.5
            
            window = 10
            successes = 0
            total = 0
            stagnation = 0
            max_stagnation = 30
            
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                d = rng.standard_normal(dim)
                x_trial = x + sigma * d
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = evaluate(x_trial)
                
                if f_trial < f:
                    x = x_trial.copy()
                    f = f_trial
                    successes += 1
                    stagnation = 0
                total += 1
                
                # Adapt sigma every window iterations
                if total >= window:
                    success_rate = successes / total
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                    total = 0
                
                sigma = np.clip(sigma, 1e-6, 5.0)
                
                if stagnation >= max_stagnation:
                    break
                
                if f_trial >= f:
                    stagnation += 1
            
            return f

        # ========================================================================
        # Budget Allocation Strategy
        # ========================================================================
        # Phase 2: BO on top 2 candidates (40% of remaining)
        bo_budget = int(remaining_budget * 0.4)
        bo_budget_per = bo_budget // 2
        
        for i in range(min(2, len(init_results))):
            if time.time() > deadline or calls_made >= budget:
                break
            x0, f0 = init_results[i]
            run_bayesian_optimization(x0, f0, bo_budget_per)
        
        # Phase 3: (1+1)-ES on top 3 candidates (50% of remaining)
        remaining_after_bo = budget - calls_made
        es_budget = int(remaining_after_bo * 0.5)
        
        # Re-sort based on current best
        # We need to track which points are good now
        # For simplicity, use the initial ranking but weight by proximity to best
        for i in range(min(3, len(init_results))):
            if time.time() > deadline or calls_made >= budget:
                break
            x0, f0 = init_results[i]
            # Allocate more to better points
            if i == 0:
                alloc = es_budget
            elif i == 1:
                alloc = es_budget // 2
            else:
                alloc = es_budget // 4
            run_11es(x0, f0, alloc)
        
        # ========================================================================
        # Phase 4: Final Polish with small sigma
        # ========================================================================
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.05
            max_polish = min(20, budget - calls_made)
            
            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 0.9
                else:
                    sigma_p *= 0.95
        
        return best_x, best_f
# EVOLVE-BLOCK-END
