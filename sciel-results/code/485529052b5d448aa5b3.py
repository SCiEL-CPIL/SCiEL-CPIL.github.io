"""Bayesian Optimization with GP surrogate and Expected Improvement acquisition
for Katsuura f23 with hidden shift/rotation/offset. Combines diverse initial
sampling with GP-guided search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as sp_minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        DEADLINE = 280.0  # safety margin before 330s kill

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            return float(func(x))

        # Data storage
        X = []
        Y = []

        def add_point(x, f):
            X.append(np.asarray(x, dtype=float).copy())
            Y.append(float(f))

        # --- Phase 1: Initial diverse sampling ---
        rng = np.random.RandomState(42)
        candidates = []
        candidates.append(np.zeros(dim))
        # Axis-aligned points
        for d in range(dim):
            for v in [-4.0, -2.0, 2.0, 4.0, -1.0, 1.0]:
                x = np.zeros(dim)
                x[d] = v
                candidates.append(x)
        # Diagonal points
        for v in [-3.0, -2.0, 2.0, 3.0]:
            candidates.append(np.full(dim, v))
        # Random points
        for _ in range(30):
            candidates.append(rng.uniform(-4.5, 4.5, size=dim))

        best_x = None
        best_f = float('inf')

        for x in candidates:
            if calls >= budget or (time.time() - start_time) > DEADLINE:
                break
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            add_point(x, fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: GP-based Bayesian Optimization ---
        # We'll implement a simple GP with RBF kernel and EI acquisition
        
        def gp_predict(X_train, y_train, X_new, length_scale=1.0, noise=1e-6):
            """Simple GP prediction with RBF kernel."""
            n = len(X_train)
            d = X_train.shape[1]
            
            # RBF kernel
            def rbf_kernel(A, B, ls):
                # Compute squared distances
                A2 = np.sum(A ** 2, axis=1, keepdims=True)
                B2 = np.sum(B ** 2, axis=1, keepdims=True)
                dist2 = A2 + B2.T - 2.0 * np.dot(A, B.T)
                np.maximum(dist2, 0.0, out=dist2)
                return np.exp(-0.5 * dist2 / (ls ** 2))
            
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
            K_new = rbf_kernel(X_train, X_new, length_scale)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                K_inv_y = alpha
            except np.linalg.LinAlgError:
                # Fallback to pinv
                K_inv_y = np.linalg.pinv(K) @ y_train
            
            # Mean prediction
            mu = K_new.T @ K_inv_y
            
            # Variance prediction
            v = np.linalg.solve(L, K_new.T)
            var = 1.0 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, f_best, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.maximum(sigma, 1e-8)
            imp = f_best - mu - xi
            z = imp / sigma
            from scipy.stats import norm
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Adaptive length scale
        length_scale = 1.5
        noise = 1e-5
        
        # Main BO loop
        no_improve_count = 0
        max_no_improve = 8
        step_size = 1.0
        
        while calls < budget and (time.time() - start_time) < DEADLINE:
            if len(X) < 3:
                # Not enough data for GP, do random exploration
                new_point = np.clip(rng.normal(best_x, 1.0, dim), -5.0, 5.0)
            else:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                f_best = np.min(Y_arr)
                
                # Generate candidate points for acquisition
                n_candidates = 200
                # Mix of local and global candidates
                local_candidates = np.clip(rng.normal(best_x, 1.0, (n_candidates // 2, dim)), -5.0, 5.0)
                global_candidates = np.clip(rng.uniform(-4.5, 4.5, (n_candidates - n_candidates // 2, dim)), -5.0, 5.0)
                cand_points = np.vstack([local_candidates, global_candidates])
                
                # GP prediction
                mu, sigma = gp_predict(X_arr, Y_arr, cand_points, length_scale, noise)
                
                # EI acquisition
                ei = expected_improvement(mu, sigma, f_best, xi=0.01)
                
                # Select best candidate
                best_idx = np.argmax(ei)
                new_point = cand_points[best_idx]
                
                # Occasionally explore more globally
                if no_improve_count > 3 and rng.random() < 0.3:
                    new_point = np.clip(rng.uniform(-4.5, 4.5, dim), -5.0, 5.0)

            new_point = np.clip(new_point, -5.0, 5.0)
            f_val = evaluate(new_point)
            add_point(new_point, f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = new_point.copy()
                no_improve_count = 0
                # Slightly increase exploration
                length_scale = min(length_scale * 1.1, 3.0)
            else:
                no_improve_count += 1
                # Decrease exploration, focus locally
                length_scale = max(length_scale * 0.9, 0.3)
            
            # If stuck, do a local refinement around best
            if no_improve_count >= max_no_improve and calls < budget:
                # Try local search with small steps
                for _ in range(5):
                    if calls >= budget or (time.time() - start_time) > DEADLINE:
                        break
                    perturbation = rng.normal(0, 0.3, dim)
                    candidate = np.clip(best_x + perturbation, -5.0, 5.0)
                    f_cand = evaluate(candidate)
                    add_point(candidate, f_cand)
                    if f_cand < best_f:
                        best_f = f_cand
                        best_x = candidate.copy()
                        no_improve_count = 0
                        break
                else:
                    # Still stuck, try a larger jump
                    new_point = np.clip(rng.normal(0, 2.0, dim), -5.0, 5.0)
                    f_val = evaluate(new_point)
                    add_point(new_point, f_val)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = new_point.copy()
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
