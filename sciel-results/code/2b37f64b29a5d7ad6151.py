"""Model-based optimizer for BBOB f14 (Different powers) using Gaussian Process surrogate."""
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def check_time():
            return time.time() > deadline

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        X = []  # list of evaluated points
        Y = []  # list of function values

        def record_eval(x, f):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            X.append(x.copy())
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget or check_time():
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            record_eval(x_clipped, f)
            return f

        # Phase 1: Initial sampling
        rng = np.random.default_rng(42)
        n_initial = min(20, budget // 5)
        
        # Latin hypercube sampling for initial points
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            f = safe_func(x)
            if f is None:
                break

        if check_time() or calls_made >= budget:
            return best_x, best_f

        # Phase 2: GP-based optimization
        # Use a simple GP with RBF kernel
        def gp_predict(X_train, y_train, X_pred, length_scale=1.0, sigma_f=1.0, sigma_n=1e-4):
            """Gaussian Process prediction."""
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_pred = np.array(X_pred)
            
            # RBF kernel
            def rbf_kernel(A, B, ls):
                A_norm = np.sum(A**2, axis=1)
                B_norm = np.sum(B**2, axis=1)
                sq_dist = A_norm[:, None] + B_norm[None, :] - 2 * A @ B.T
                sq_dist = np.maximum(sq_dist, 0)
                return sigma_f**2 * np.exp(-sq_dist / (2 * ls**2))
            
            K = rbf_kernel(X_train, X_train, length_scale)
            K += sigma_n**2 * np.eye(len(X_train))
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter for numerical stability
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback: return mean and high uncertainty
                    return np.mean(y_train) * np.ones(len(X_pred)), np.std(y_train) * np.ones(len(X_pred))
            
            Ks = rbf_kernel(X_pred, X_train, length_scale)
            
            # Mean prediction
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mu = Ks @ alpha
            
            # Variance prediction
            v = np.linalg.solve(L, Ks.T)
            var = sigma_f**2 - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8)
            
            return mu, np.sqrt(var)

        # Adaptive length scale based on data spread
        def estimate_length_scale(X_data):
            if len(X_data) < 2:
                return 1.0
            X_arr = np.array(X_data)
            # Average pairwise distance
            from scipy.spatial.distance import cdist
            dists = cdist(X_arr, X_arr)
            upper = np.triu(dists, k=1)
            mean_dist = np.mean(upper)
            return max(0.5, min(5.0, mean_dist))

        # Acquisition function: Expected Improvement
        def expected_improvement(mu, sigma, best):
            if sigma < 1e-10:
                return 0.0
            z = (best - mu) / sigma
            ei = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Main optimization loop
        max_gp_iters = budget - calls_made - 50  # Reserve some budget for final local search
        gp_iters_done = 0
        
        while gp_iters_done < max_gp_iters and not check_time() and calls_made < budget:
            # Update GP model
            if len(X) >= 2:
                length_scale = estimate_length_scale(X)
                # Scale sigma_f to data range
                y_range = max(np.max(Y) - np.min(Y), 1e-3)
                sigma_f = y_range / 2 + 1e-3
                
                # Predict on a candidate set
                # Generate candidates: mix of random and around current best
                n_random = 50
                n_around_best = 50
                n_total = n_random + n_around_best
                
                candidates = []
                # Random candidates
                for _ in range(n_random):
                    candidates.append(rng.uniform(-5, 5, dim))
                # Candidates around best
                for _ in range(n_around_best):
                    offset = rng.normal(0, 0.5, dim)
                    candidates.append(np.clip(best_x + offset, -5, 5))
                
                candidates = np.array(candidates)
                mu, sigma = gp_predict(X, Y, candidates, length_scale, sigma_f)
                
                # Compute EI
                ei = expected_improvement(mu, sigma, best_f)
                
                # Select best candidate
                best_idx = np.argmax(ei)
                x_next = candidates[best_idx]
            else:
                # Not enough data, use random
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate
            f = safe_func(x_next)
            if f is None:
                break
            gp_iters_done += 1

        # Phase 3: Local refinement using finite-difference BFGS
        if not check_time() and calls_made < budget - 10:
            # Use the best point found so far
            x = best_x.copy()
            fx = best_f
            eps = 1e-4
            H = np.eye(dim)
            
            # Compute initial gradient
            grad = np.zeros(dim)
            for j in range(dim):
                x_plus = x.copy()
                x_plus[j] += eps
                fp = safe_func(x_plus)
                if fp is None:
                    break
                grad[j] = (fp - fx) / eps
            else:
                # BFGS iterations
                max_local_iter = min(budget - calls_made, 100)
                for it in range(max_local_iter):
                    if check_time() or calls_made >= budget:
                        break
                    
                    if np.linalg.norm(grad) < 1e-6:
                        break
                    
                    # Search direction
                    d = H @ grad
                    dn = np.linalg.norm(d)
                    if dn < 1e-12:
                        break
                    d = -d / dn
                    
                    # Line search
                    alpha = 1.0
                    success = False
                    for _ in range(10):
                        x_try = x + alpha * d
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_func(x_try)
                        if f_try is None:
                            break
                        if f_try < fx - 1e-8:
                            success = True
                            break
                        alpha *= 0.5
                    
                    if not success:
                        break
                    
                    # Update
                    s = x_try - x
                    ys = f_try - fx
                    x, fx = x_try, f_try
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Update Hessian approximation
                    Sy = s @ ys
                    if Sy > 1e-10:
                        Hs = H @ s
                        H = H - (np.outer(Hs, s) + np.outer(s, Hs)) / Sy + \
                            (np.outer(s, s) * (Sy + s @ (H @ s))) / (Sy * Sy)
                        H = 0.5 * (H + H.T)
                    
                    # Recompute gradient
                    grad_new = np.zeros(dim)
                    for j in range(dim):
                        x_plus = x.copy()
                        x_plus[j] += eps
                        fp = safe_func(x_plus)
                        if fp is None:
                            break
                        grad_new[j] = (fp - fx) / eps
                    else:
                        grad = grad_new

        return best_x, best_f
