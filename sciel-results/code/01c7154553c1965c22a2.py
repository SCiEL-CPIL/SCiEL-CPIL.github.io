"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Two-phase Surrogate-Assisted Optimizer for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        def time_left():
            return deadline - time.time()
            
        def elapsed():
            return time.time() - start_time

        n = self.dim
        budget = self.budget
        call_count = [0]
        
        def safe_call(x):
            if call_count[0] >= budget:
                return float('inf')
            call_count[0] += 1
            if elapsed() > deadline - 5:
                return float('inf')
            return func(np.clip(x, -5, 5))
        
        X_train = []
        y_train = []
        
        rng = np.random.default_rng(42)
        
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-sq_dists / (2 * length_scale**2))

        def latin_hypercube(n_points, dim, seed=42):
            rng_lh = np.random.default_rng(seed)
            points = np.zeros((n_points, dim))
            for d in range(dim):
                perm = rng_lh.permutation(n_points)
                points[:, d] = (perm + rng_lh.uniform(size=n_points)) / n_points
            points = points * 10.0 - 5.0
            return points

        # Phase 1: GP-based exploration with ~30 calls
        n_init = 12
        X_init = latin_hypercube(n_init, n)
        
        for x in X_init:
            if elapsed() > deadline - 20:
                break
            f_val = safe_call(x)
            if np.isinf(f_val):
                break
            X_train.append(x)
            y_train.append(f_val)
            
        if not X_train:
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            return x0, f0

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        noise_var = 1e-8
        signal_var = 1.0
        
        def fit_gp(X_tr, y_tr, length_scale):
            K_train = rbf_kernel(X_tr, X_tr, length_scale, signal_var) + noise_var * np.eye(len(X_tr))
            K_train += 1e-6 * np.eye(len(X_tr))
            try:
                L_train = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_tr))
                return L_train, alpha, True
            except:
                return None, None, False

        def gp_mean_fn(x_pred, X_tr, L_tr, alpha, length_scale):
            x_pred = x_pred.reshape(1, -1)
            K_s = rbf_kernel(x_pred, X_tr, length_scale, signal_var)
            mu = K_s @ alpha
            return mu[0]

        # Adaptive length scale: estimate from data spread
        if len(X_train) >= 3:
            # Use median pairwise distance as a rough scale
            from scipy.spatial.distance import pdist
            dists = pdist(X_train)
            length_scale = max(np.median(dists) * 0.5, 0.3)
        else:
            length_scale = 1.0

        L_train, alpha, gp_ok = fit_gp(X_train, y_train, length_scale)
        
        if gp_ok:
            # Multiple rounds of GP-guided acquisition
            n_gp_rounds = 8
            for r in range(n_gp_rounds):
                if elapsed() > deadline - 150:
                    break
                if call_count[0] >= budget - 50:
                    break
                    
                current_best_idx = np.argmin(y_train)
                candidates = [X_train[current_best_idx].copy()]
                # Add random candidates spread across the domain
                for _ in range(4):
                    candidates.append(rng.uniform(-4, 4, size=n))
                # Add candidates near the best point
                for _ in range(2):
                    candidates.append(X_train[current_best_idx] + rng.normal(0, 0.5, size=n))
                
                best_gp_x = None
                best_gp_val = float('inf')
                
                for c in candidates:
                    c_clipped = np.clip(c, -5, 5)
                    try:
                        res = minimize(
                            lambda x: gp_mean_fn(x, X_train, L_train, alpha, length_scale),
                            c_clipped, method='Nelder-Mead',
                            options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-6}
                        )
                        if res.fun < best_gp_val:
                            best_gp_val = res.fun
                            best_gp_x = np.clip(res.x, -5, 5)
                    except:
                        continue
                
                if best_gp_x is not None:
                    f_eval = safe_call(best_gp_x)
                    if not np.isinf(f_eval):
                        X_train = np.vstack([X_train, best_gp_x])
                        y_train = np.append(y_train, f_eval)
                        
                        # Re-fit GP with updated data, possibly adjusting length_scale
                        # Shrink length_scale slightly as we refine
                        length_scale = max(length_scale * 0.85, 0.2)
                        L_train, alpha, gp_ok = fit_gp(X_train, y_train, length_scale)
                        if not gp_ok:
                            break

        # End Phase 1
        best_idx = np.argmin(y_train)
        x_start = X_train[best_idx]
        f_start = y_train[best_idx]
        
        # Phase 2: Local Refinement with L-BFGS-B
        bounds = [(-5, 5)] * n
        
        def objective(x):
            if elapsed() > deadline - 5:
                return float('inf')
            if call_count[0] >= budget:
                return float('inf')
            call_count[0] += 1
            return func(np.clip(x, -5, 5))
        
        best_x_final = x_start.copy()
        best_f_final = f_start
        
        # Primary L-BFGS-B from GP-selected start
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_f_final:
                best_f_final = res.fun
                best_x_final = np.clip(res.x, -5, 5)
        except Exception:
            pass
            
        # Multiple restarts with decreasing perturbation scales
        if elapsed() < deadline - 30 and call_count[0] < budget - 20:
            scales = [1.0, 0.5, 0.3, 0.15, 0.08, 0.04]
            for scale in scales:
                if elapsed() > deadline - 15:
                    break
                if call_count[0] >= budget - 15:
                    break
                for attempt in range(2):
                    if elapsed() > deadline - 10:
                        break
                    if call_count[0] >= budget - 10:
                        break
                    perturbation = rng.normal(0, scale, size=n)
                    x_trial = np.clip(best_x_final + perturbation, -5, 5)
                    try:
                        res = minimize(objective, x_trial, method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                        if res.fun < best_f_final:
                            best_f_final = res.fun
                            best_x_final = np.clip(res.x, -5, 5)
                    except:
                        continue
                        
        # Final polish: very tight tolerance from the best point
        if elapsed() < deadline - 10 and call_count[0] < budget - 5:
            try:
                res = minimize(objective, best_x_final, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < best_f_final:
                    best_f_final = res.fun
                    best_x_final = np.clip(res.x, -5, 5)
            except:
                pass
                    
        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
