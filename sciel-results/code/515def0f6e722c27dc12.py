"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization (GP) + Nelder-Mead polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start exploration - sample diverse points
        n_initial = 15
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-4, 4, n)
            elif i == 2:
                x = rng.uniform(-2, 2, n)
            elif i == 3:
                x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Bayesian Optimization with Gaussian Process surrogate
        # Collect initial observations
        X_obs = np.array([c[1] for c in candidates])
        y_obs = np.array([c[0] for c in candidates])
        
        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6
        
        def rbf_kernel(X1, X2, ls):
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-sq_dist / (2 * ls**2))
        
        def gp_predict(X_new, X_train, y_train, ls, sig_var, noise_var):
            K_train = rbf_kernel(X_train, X_train, ls) + noise_var * np.eye(len(X_train))
            K_cross = rbf_kernel(X_new, X_train, ls)
            K_new = rbf_kernel(X_new, X_new, ls)
            
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                v = np.linalg.solve(L, K_cross.T)
                mu = K_cross @ alpha
                var = sig_var + np.sum(K_new, axis=1) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                K_reg = K_train + 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K_reg)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                    v = np.linalg.solve(L, K_cross.T)
                    mu = K_cross @ alpha
                    var = sig_var + np.sum(K_new, axis=1) - np.sum(v**2, axis=0)
                    var = np.maximum(var, 1e-10)
                except np.linalg.LinAlgError:
                    mu = np.mean(y_train) * np.ones(len(X_new))
                    var = np.var(y_train) * np.ones(len(X_new))
            
            return mu, var
        
        # EI acquisition function
        def expected_improvement(mu, var, f_best, xi=0.01):
            improvement = f_best - mu - xi
            std = np.sqrt(var)
            z = improvement / (std + 1e-10)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(std > 1e-8, ei, 0.0)
            return ei
        
        # BO loop
        bo_calls = 0
        max_bo_calls = min(budget - calls, 200)
        
        for _ in range(max_bo_calls):
            if calls >= budget or not time_ok():
                break
            
            # Sample candidate points for acquisition
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, n))
            
            # Predict on candidates
            mu, var = gp_predict(X_cand, X_obs, y_obs, length_scale, signal_var, noise_var)
            
            # Compute EI
            ei = expected_improvement(mu, var, best_f, xi=0.01)
            
            # Select point with max EI
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            # Evaluate
            fx = evaluate(x_next)
            if fx is None:
                break
            
            # Update observations
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, fx)
            bo_calls += 1
            
            # Adapt length scale based on progress
            if bo_calls % 20 == 0 and bo_calls > 0:
                # Estimate typical distance between good points
                if len(X_obs) > 2:
                    # Use distances between points with similar function values
                    y_sorted = np.sort(y_obs)
                    mid_idx = len(y_sorted) // 2
                    good_points = X_obs[np.argsort(y_obs)[:mid_idx]]
                    if len(good_points) > 1:
                        dists = []
                        for i in range(len(good_points)):
                            for j in range(i+1, len(good_points)):
                                dists.append(np.linalg.norm(good_points[i] - good_points[j]))
                        if dists:
                            length_scale = max(0.1, np.median(dists))
        
        # Phase 3: Nelder-Mead polish on best point
        if best_x is not None and calls < budget and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            # Build simplex
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                if x0[j] < 4.0:
                    delta = 0.3
                else:
                    delta = -0.3
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            fvals = [f0]
            for pt in simplex[1:]:
                if calls >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)
            
            if len(fvals) != len(simplex):
                return best_x, best_f
            
            # NM iterations
            max_nm_iter = max(0, budget - calls)
            for _ in range(max_nm_iter):
                if calls >= budget or not time_ok():
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = evaluate(xr)
                if fr is None:
                    break
                
                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                        fvals_new = [fvals[0]]
                        for pt in simplex[1:]:
                            if calls >= budget or not time_ok():
                                break
                            fx = evaluate(pt)
                            if fx is None:
                                break
                            fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
