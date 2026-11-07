"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Surrogate-assisted Bayesian optimization with GP and EI acquisition for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget
        X_list = []
        y_list = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = float(func(x))
            eval_count += 1
            X_list.append(x.copy())
            y_list.append(fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        def gp_predict(X_train, y_train, X_test, length_scale=1.5, noise=1e-6):
            """Simple GP prediction using RBF kernel."""
            n = len(y_train)
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            
            def rbf_kernel(A, B, ls):
                A2 = np.sum(A**2, axis=1)[:, None]
                B2 = np.sum(B**2, axis=1)[None, :]
                D2 = np.maximum(A2 + B2 - 2 * A @ B.T, 0)
                return np.exp(-D2 / (2 * ls**2))
            
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
            K_star = rbf_kernel(X_train, X_test, length_scale)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - np.mean(y_train)))
                mu = np.mean(y_train) + K_star.T @ alpha
                v = np.linalg.solve(L, K_star)
                var = np.var(y_train) + length_scale**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
            except np.linalg.LinAlgError:
                K_reg = K + 1e-6 * np.eye(n)
                alpha = np.linalg.solve(K_reg, y_train - np.mean(y_train))
                mu = np.mean(y_train) + K_star.T @ alpha
                v = np.linalg.solve(K_reg, K_star)
                var = np.var(y_train) + length_scale**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
            
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_val, xi=0.01):
            """Compute EI acquisition values."""
            sigma = np.maximum(sigma, 1e-12)
            imp = best_val - mu - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        def optimize_acquisition(X_train, y_train, n_restarts=5, max_iter=50):
            """Optimize EI acquisition function using multiple random restarts."""
            best_val = min(y_list)
            best_x_acq = None
            best_ei = -np.inf
            
            starts = [np.random.uniform(-5, 5, size=dim) for _ in range(n_restarts)]
            if best_x is not None:
                starts.append(best_x + np.random.normal(0, 0.5, size=dim))
                starts.append(best_x + np.random.normal(0, 1.0, size=dim))
            
            for start in starts:
                start = np.clip(start, -5, 5)
                try:
                    res = minimize(
                        lambda x: -expected_improvement(*gp_predict(X_train, y_train, x.reshape(1, -1)), best_val)[0][0],
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x_acq = res.x
                except Exception:
                    continue
            
            if best_x_acq is None:
                best_x_acq = np.random.uniform(-5, 5, size=dim)
            
            return np.clip(best_x_acq, -5, 5)

        # Phase 1: Initial diverse sampling
        n_initial = 25
        np.random.seed(42)
        
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        initial_points = sampler.random(n=n_initial) * 10.0 - 5.0
        
        strategic = [
            np.zeros(dim),
            np.ones(dim),
            -np.ones(dim),
            np.array([1.0] * dim),
            np.array([0.5] * dim),
            np.array([2.0] * dim),
        ]
        
        all_initial = list(initial_points) + strategic
        np.random.shuffle(all_initial)
        
        for p in all_initial:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
                X_list.append(best_x.copy())
                y_list.append(best_f)
            except Exception:
                pass

        # Phase 2: Bayesian Optimization loop
        n_bo_iter = 0
        max_bo_iter = 200
        
        while eval_count < budget - 30 and n_bo_iter < max_bo_iter:
            if time.time() > deadline - 5:
                break
            
            n_bo_iter += 1
            
            try:
                x_next = optimize_acquisition(np.array(X_list), np.array(y_list), 
                                              n_restarts=5, max_iter=50)
            except Exception:
                x_next = np.random.uniform(-5, 5, size=dim)
            
            try:
                f_next = wrapped_func(x_next)
            except Exception:
                break
            
            if best_f < 1e-6:
                break
        
        # Phase 3: Local polish with L-BFGS-B from best point
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 4: Nelder-Mead polish
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': remaining, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 5: Final L-BFGS-B pass
        remaining = budget - eval_count
        if remaining > 3 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
