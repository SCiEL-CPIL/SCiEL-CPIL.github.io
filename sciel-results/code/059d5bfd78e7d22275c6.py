import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with GP surrogate and EI acquisition for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Storage for observations
        X_obs = []  # List of points
        Y_obs = []  # List of function values
        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def add_observation(x, f):
            nonlocal best_x, best_f, used_calls
            X_obs.append(x.copy())
            Y_obs.append(f)
            used_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial random sampling to build initial GP training set
        n_init = min(30, budget)
        for _ in range(n_init):
            if used_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            add_observation(x, f)

        if best_x is None:
            return np.zeros(d), best_f

        # GP kernel: Matern 5/2 with ARD
        def gp_predict(X_new, X_train, Y_train, length_scales, variance, noise):
            """GP prediction with Matern 5/2 kernel."""
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            X_new = np.array(X_new)
            
            n_train = len(X_train)
            n_new = len(X_new)
            
            # Compute kernel matrix K(X_train, X_train)
            K = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(i+1, n_train):
                    diff = (X_train[i] - X_train[j]) / length_scales
                    r = np.sqrt(np.sum(diff**2))
                    if r < 1e-10:
                        k_val = variance
                    else:
                        k_val = variance * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)
                    K[i,j] = k_val
                    K[j,i] = k_val
                K[i,i] = variance + noise
            
            # Compute kernel vector K(X_new, X_train)
            K_s = np.zeros((n_new, n_train))
            for i in range(n_new):
                for j in range(n_train):
                    diff = (X_new[i] - X_train[j]) / length_scales
                    r = np.sqrt(np.sum(diff**2))
                    if r < 1e-10:
                        k_val = variance
                    else:
                        k_val = variance * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)
                    K_s[i,j] = k_val
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n_train))
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            cov = variance - v.T @ v
            std = np.sqrt(np.maximum(cov, 1e-12))
            
            return mu, std

        # Hyperparameters
        length_scales = np.ones(d) * 1.0
        variance = 1.0
        noise = 1e-8

        # Phase 2: Bayesian Optimization loop
        n_bo_iter = min(budget - used_calls, 200)
        
        for bo_iter in range(n_bo_iter):
            if used_calls >= budget or check_time():
                break
            
            remaining = budget - used_calls
            if remaining < 5:
                break
            
            # Update hyperparameters occasionally (simple approach: keep fixed or tune)
            if bo_iter % 10 == 0 and len(X_obs) > 10:
                # Simple hyperparameter update based on spread of data
                X_arr = np.array(X_obs)
                Y_arr = np.array(Y_obs)
                if np.std(Y_arr) > 1e-10:
                    variance = np.var(Y_arr) + 1e-6
                # Adjust length scales based on data spread
                for i in range(d):
                    spread = np.max(X_arr[:, i]) - np.min(X_arr[:, i])
                    if spread > 1e-6:
                        length_scales[i] = max(spread / 3, 0.1)
            
            # Generate candidate points for EI maximization
            n_candidates = min(50, remaining - 1)
            
            # Mix of random and locally perturbed candidates
            candidates = []
            
            # Random candidates
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, d))
            
            # Perturbations around best point
            for scale in [0.5, 0.2, 0.1, 0.05]:
                for _ in range(n_candidates // 4):
                    x = best_x + scale * rng.standard_normal(d)
                    x = np.clip(x, -5, 5)
                    candidates.append(x)
            
            # If we have more candidates than needed, sample
            if len(candidates) > n_candidates:
                idx = rng.choice(len(candidates), n_candidates, replace=False)
                candidates = [candidates[i] for i in idx]
            
            # Compute EI for each candidate
            X_train = np.array(X_obs)
            Y_train = np.array(Y_obs)
            Y_min = np.min(Y_train)
            
            best_ei = -np.inf
            best_candidate = None
            
            for cand in candidates:
                mu, std = gp_predict(cand.reshape(1, -1), X_train, Y_train, 
                                      length_scales, variance, noise)
                mu = mu[0]
                std = std[0]
                
                if std < 1e-8:
                    ei = 0.0
                else:
                    z = (Y_min - mu) / std
                    ei = (Y_min - mu) * norm.cdf(z) + std * norm.pdf(z)
                
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = cand
            
            if best_candidate is None:
                best_candidate = rng.uniform(-5, 5, d)
            
            # Evaluate the selected point
            f = func(best_candidate)
            add_observation(best_candidate, f)
            
            # If we still have budget, do a few local refinements
            if used_calls < budget and not check_time():
                for scale in [0.1, 0.05, 0.01]:
                    if used_calls >= budget or check_time():
                        break
                    for _ in range(min(3, budget - used_calls)):
                        x = best_x + scale * rng.standard_normal(d)
                        x = np.clip(x, -5, 5)
                        f_local = func(x)
                        add_observation(x, f_local)

        # Phase 3: Final local refinement with Nelder-Mead
        if best_x is not None and used_calls < budget and not check_time():
            remaining = budget - used_calls
            if remaining >= 20:
                max_local_calls = min(remaining - 5, 100)
                calls_made = [0]
                
                def wrapped_func(x, calls_made=calls_made, max_local_calls=max_local_calls):
                    calls_made[0] += 1
                    if calls_made[0] > max_local_calls or check_time():
                        return 1e10
                    return func(x)

                try:
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10,
                                            'maxiter': max_local_calls,
                                            'adaptive': True})
                    used_calls += calls_made[0]
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
