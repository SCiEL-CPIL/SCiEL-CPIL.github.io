import numpy as np
import time
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        # Data storage
        X = []
        Y = []
        fbest = np.inf
        x_best = np.zeros(n)

        def evaluate(x):
            nonlocal calls, fbest, x_best
            if calls >= budget or not time_ok():
                return fbest
            x_eval = np.clip(x, -5, 5)
            fx = func(x_eval)
            calls += 1
            X.append(x_eval.copy())
            Y.append(fx)
            if fx < fbest:
                fbest = fx
                x_best = x_eval.copy()
            return fx

        # Phase 1: Initial Sampling
        # Use a mix of center and random points to cover the space
        n_initial = 10
        for _ in range(n_initial):
            x_init = rng.uniform(-4, 4, n)
            evaluate(x_init)
        
        # Also evaluate the center
        if calls < budget:
            evaluate(np.zeros(n))

        # Phase 2: GP Surrogate Search
        # We will run this loop until we have used a significant portion of the budget
        # or until the GP suggests we are close to the optimum.
        
        # Parameters for GP
        # We use a simple RBF kernel. In 5D, this is computationally feasible for ~100-200 points.
        
        def compute_kernel(X1, X2):
            # RBF kernel
            # sigma_l = 1.0, sigma_f = 1.0
            # K(x, y) = sigma_f^2 * exp(-0.5 * ||x-y||^2 / sigma_l^2)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return np.exp(-0.5 * sq_dist)

        def compute_gp(X_train, y_train, X_test, alpha=None):
            """
            Compute GP mean and variance.
            Returns mean, std.
            """
            if alpha is None:
                K = compute_kernel(X_train, X_train)
                # Add jitter for numerical stability
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K += 1e-4 * np.eye(len(X_train))
                    L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            K_star = compute_kernel(X_test, X_train)
            mean = K_star @ alpha
            K_ss = compute_kernel(X_test, X_test)
            v = np.linalg.solve(L, K_star.T)
            var = 1.0 - np.sum(v ** 2, axis=1) # sigma_f^2 = 1
            var = np.clip(var, 1e-8, None)
            std = np.sqrt(var)
            return mean, std, alpha

        # Expected Improvement
        def expected_improvement(mean, std, y_best):
            # EI = (y_best - mean) * Phi((y_best - mean)/std) + std * phi((y_best - mean)/std)
            # For minimization, we want to improve by (y_best - y_new)
            # Let z = (y_best - mean) / std
            z = (y_best - mean) / (std + 1e-8)
            ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Loop for GP guided search
        # We reserve some budget for local refinement
        gp_budget_limit = int(budget * 0.6)
        
        while calls < gp_budget_limit and time_ok():
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            y_min = np.min(Y_arr)
            
            # Compute GP on a grid or use random candidates?
            # In 5D, a full grid is too large. We use a set of candidate points.
            # Candidates: 
            # 1. Best point + small perturbations
            # 2. Random points in the box
            # 3. Points along the line connecting best and second best?
            
            n_candidates = 200
            candidates = []
            
            # 1. Perturbations around best
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(10):
                    x_cand = x_best + scale * rng.standard_normal(n)
                    candidates.append(np.clip(x_cand, -5, 5))
            
            # 2. Random points
            for _ in range(n_candidates - len(candidates)):
                x_cand = rng.uniform(-4, 4, n)
                candidates.append(x_cand)
                
            candidates = np.array(candidates)
            
            # Compute GP mean and std for candidates
            try:
                mean, std, alpha = compute_gp(X_arr, Y_arr, candidates)
            except Exception:
                # Fallback to random if GP fails
                x_new = rng.uniform(-4, 4, n)
                evaluate(x_new)
                continue

            # Compute EI
            ei = expected_improvement(mean, std, y_min)
            
            # Select best candidate
            idx_best = np.argmax(ei)
            x_new = candidates[idx_best]
            
            # Evaluate new point
            evaluate(x_new)

        # Phase 3: Local Refinement
        # Use Nelder-Mead from the best point found by GP
        if calls < budget and time_ok():
            try:
                def wrapped_func_nm(x):
                    nonlocal calls
                    if calls >= budget or not time_ok():
                        return fbest
                    x_eval = np.clip(x, -5, 5)
                    fx = func(x_eval)
                    calls += 1
                    # We don't update X, Y here to keep GP clean, 
                    # but we do update fbest/x_best if we want, 
                    # but scipy handles the best internally.
                    # However, for the final result, we track it.
                    if fx < fbest:
                        nonlocal x_best
                        x_best = x_eval.copy()
                        fbest = fx
                    return fx

                # Use a tight tolerance
                result = scipy.optimize.minimize(
                    wrapped_func_nm, x_best, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 4: Final Polish
        # If we still have budget, do a few more random steps near the best
        if calls < budget and time_ok():
            for _ in range(min(10, budget - calls)):
                x_try = x_best + 0.01 * rng.standard_normal(n)
                x_try = np.clip(x_try, -5, 5)
                fx = func(x_try)
                calls += 1
                if fx < fbest:
                    fbest = fx
                    x_best = x_try.copy()

        return x_best, fbest
