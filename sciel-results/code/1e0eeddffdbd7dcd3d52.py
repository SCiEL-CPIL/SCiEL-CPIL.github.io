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
        
        def safe_func(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return float('inf')
            return val

        X_train = []
        y_train = []
        
        rng = np.random.default_rng(42)
        
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-sq_dists / (2 * length_scale**2))

        gp_budget = 50
        
        def latin_hypercube(n_points, dim, seed=42):
            rng_lh = np.random.default_rng(seed)
            points = np.zeros((n_points, dim))
            for d in range(dim):
                perm = rng_lh.permutation(n_points)
                points[:, d] = (perm + rng_lh.uniform(size=n_points)) / n_points
            points = points * 10.0 - 5.0
            return points

        n_init = 10
        X_init = latin_hypercube(n_init, n)
        
        for x in X_init:
            if call_count[0] >= gp_budget or elapsed() > deadline - 30:
                break
            f_val = safe_func(x)
            if np.isfinite(f_val):
                X_train.append(x)
                y_train.append(f_val)
            
        if not X_train:
            x0 = np.zeros(n)
            f0 = safe_func(x0)
            return x0, f0

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Adaptive length scale based on data spread
        if len(X_train) > 1:
            diffs = np.diff(X_train, axis=0)
            median_dist = np.median(np.sqrt(np.sum(diffs**2, axis=1)))
            length_scale = max(0.5, min(2.0, median_dist))
        else:
            length_scale = 1.0
        noise_var = 1e-6
        signal_var = 1.0
        
        def fit_gp():
            nonlocal length_scale, noise_var, signal_var
            K_train = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
            K_train += 1e-6 * np.eye(len(X_train))
            try:
                L_train = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
                return L_train, alpha
            except np.linalg.LinAlgError:
                return None, None
        
        L_train, alpha = fit_gp()
        
        if L_train is not None:
            def gp_mean(x_pred):
                x_pred = np.atleast_2d(x_pred)
                K_s = rbf_kernel(x_pred, X_train, length_scale, signal_var)
                mu = K_s @ alpha
                return mu[0]
            
            # Use remaining GP budget for iterative acquisition
            gp_rounds = min(39, gp_budget - call_count[0])
            for round_idx in range(gp_rounds):
                if call_count[0] >= gp_budget or elapsed() > deadline - 120:
                    break
                
                current_best_idx = np.argmin(y_train)
                candidates = [X_train[current_best_idx]]
                # Add diverse random starts
                for _ in range(4):
                    candidates.append(rng.uniform(-4, 4, size=n))
                
                best_gp_x = None
                best_gp_val = float('inf')
                
                for c in candidates:
                    try:
                        res = minimize(gp_mean, c, method='Nelder-Mead', 
                                       options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-5})
                        if res.fun < best_gp_val:
                            best_gp_val = res.fun
                            best_gp_x = res.x
                    except Exception:
                        continue
                
                if best_gp_x is not None:
                    x_eval = np.clip(best_gp_x, -5, 5)
                    f_eval = safe_func(x_eval)
                    if np.isfinite(f_eval):
                        X_train = np.vstack([X_train, x_eval])
                        y_train = np.append(y_train, f_eval)
                        L_train, alpha = fit_gp()
                        if L_train is None:
                            break

        # End Phase 1
        best_idx = np.argmin(y_train)
        x_start = X_train[best_idx]
        f_start = y_train[best_idx]
        
        # Phase 2: Local Refinement with L-BFGS-B
        bounds = [(-5, 5)] * n
        
        def objective(x):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                return float('inf')
            return safe_func(x)
            
        best_x_final = x_start.copy()
        best_f_final = f_start
        
        # Primary L-BFGS-B run
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_f_final:
                best_f_final = res.fun
                best_x_final = res.x.copy()
        except Exception:
            pass
        
        # Multi-restart from perturbed points if budget allows
        if call_count[0] < budget - 50 and elapsed() < deadline - 30:
            for _ in range(8):
                if call_count[0] >= budget - 20 or elapsed() > deadline - 15:
                    break
                perturbation = rng.normal(0, 0.3, size=n)
                x_trial = np.clip(best_x_final + perturbation, -5, 5)
                try:
                    res = minimize(objective, x_trial, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res.fun < best_f_final:
                        best_f_final = res.fun
                        best_x_final = res.x.copy()
                except Exception:
                    continue
        
        # Final high-precision polish pass
        if call_count[0] < budget - 10 and elapsed() < deadline - 10:
            try:
                res = minimize(objective, best_x_final, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < best_f_final:
                    best_f_final = res.fun
                    best_x_final = res.x.copy()
            except Exception:
                pass
                
        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
