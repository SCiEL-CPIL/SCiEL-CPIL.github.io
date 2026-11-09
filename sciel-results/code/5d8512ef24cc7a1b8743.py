"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with surrogate-assisted global exploration phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 330s kill

        def time_left():
            return time.time() < deadline

        # --- Phase 1: Surrogate-assisted global exploration ---
        # Use a simple kriging/GP surrogate to guide initial search.
        # This helps find a good starting region for CMA-ES.
        
        # Initial random sampling for surrogate
        n_init = 30
        X_init = np.random.uniform(-4.5, 4.5, size=(n_init, n))
        X_init = np.clip(X_init, -5, 5)
        
        # Evaluate initial points
        X_list = [X_init[i].copy() for i in range(n_init)]
        F_list = [func(X_list[i]) for i in range(n_init)]
        
        n_calls = n_init
        best_f = min(F_list)
        best_x = X_list[int(np.argmin(F_list))].copy()
        
        # Fit a simple quadratic surrogate (fast and effective for 5D)
        # Features: x, x^2, x_i*x_j for i<j
        def make_features(x):
            # x: (n,) -> (n + n*(n+1)/2,)
            feats = [x]
            for i in range(n):
                feats.append([x[i]**2])
                for j in range(i+1, n):
                    feats.append([x[i]*x[j]])
            return np.concatenate(feats)
        
        def fit_surrogate(X, F):
            """Fit a quadratic polynomial surrogate."""
            A = np.array([make_features(x) for x in X])
            A = np.hstack([A, np.ones((len(X), 1))])
            # Ridge regression for stability
            lam = 0.1
            try:
                coef, _, _, _ = np.linalg.lstsq(A + lam*np.eye(A.shape[1]), F, rcond=None)
            except:
                coef = np.zeros(A.shape[1])
            return coef
        
        def surrogate_eval(x, coef):
            feat = make_features(x)
            feat = np.append(feat, 1.0)
            return float(feat @ coef)
        
        # Iterative surrogate-guided search
        n_surrogate_iters = 5
        for it in range(n_surrogate_iters):
            if n_calls >= budget or not time_left():
                break
            
            coef = fit_surrogate(X_list, F_list)
            
            # Generate candidate points: grid + random
            candidates = []
            # Local refinement around best
            for dx in np.linspace(-1, 1, 3):
                for dy in np.linspace(-1, 1, 3):
                    for dz in np.linspace(-1, 1, 3):
                        cand = best_x.copy()
                        cand[0] += dx
                        cand[1] += dy
                        cand[2] += dz
                        if len(cand) > 3:
                            cand[3] += (dx + dy) / 2
                            cand[4] += (dz + dx) / 2
                        candidates.append(np.clip(cand, -5, 5))
            
            # Random points
            candidates.extend(np.random.uniform(-4.5, 4.5, size=(20, n)))
            candidates = [np.clip(c, -5, 5) for c in candidates]
            
            # Score candidates by surrogate
            scores = [surrogate_eval(c, coef) for c in candidates]
            scores_arr = np.array(scores)
            
            # Select top 5 diverse candidates
            # Simple diversity: pick from different regions
            selected = []
            for idx in np.argsort(scores_arr)[:10]:
                c = candidates[idx]
                # Check if sufficiently different from selected
                is_diverse = True
                for s in selected:
                    if np.linalg.norm(c - s) < 0.5:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append(c)
                if len(selected) >= 5:
                    break
            
            # If not enough diverse, fill with best remaining
            if len(selected) < 5:
                for idx in np.argsort(scores_arr):
                    c = candidates[idx]
                    if c not in selected:
                        selected.append(c)
                    if len(selected) >= 5:
                        break
            
            # Evaluate selected candidates
            for c in selected:
                if n_calls >= budget or not time_left():
                    break
                f = func(c)
                n_calls += 1
                X_list.append(c)
                F_list.append(f)
                if f < best_f:
                    best_f = f
                    best_x = c.copy()
        
        # --- Phase 2: CMA-ES local optimization from best point ---
        m = best_x.copy()
        sigma = 1.0  # initial step size
        
        # CMA-ES parameters
        lambda_ = 12  # population size
        mu = lambda_ // 2  # number of parents
        # weights (logarithmic)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size adaptation
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        # covariance adaptation
        cmu = min(1 - 1 / (n * n), 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        c1 = (5 - n + mueff) / ((n + 2) ** 3 + mueff)
        c1 = max(c1, 0.1 * cmu)
        cov_update = c1 + 2 * cmu * (1 - c1 - 0.5 * c1)
        
        # mean
        m = m.copy()
        C = np.eye(n)
        Cinv_sqrt = np.eye(n)
        
        def update_eig():
            nonlocal Cinv_sqrt
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-20)
            Cinv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
        
        update_eig()
        
        # path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Main CMA-ES loop
        while n_calls < budget and time_left():
            # Generate population
            ys = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                y = m + sigma * (Cinv_sqrt @ z)
                ys[i] = np.clip(y, -5, 5)
                n_calls += 1
                if n_calls >= budget:
                    break
            
            if n_calls >= budget:
                break
            
            # Evaluate population
            fs = np.empty(lambda_)
            for i in range(lambda_):
                fs[i] = func(ys[i])
                n_calls += 1
                if not time_left():
                    fs[i:] = np.inf
                    break
            
            # Track best
            fmin = np.min(fs)
            if fmin < best_f:
                best_f = fmin
                idx_best = np.argmin(fs)
                best_x = ys[idx_best].copy()
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            m_new = m + sigma * np.sum(w[:, None] * (ys_sorted[:mu] - m)[:, :mu].T, axis=1)
            m_new = np.clip(m_new, -5, 5)
            
            # Update sigma (path)
            y_sigma = Cinv_sqrt.T @ (m_new - m) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) - 1 / (24 * n ** 2))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (budget - n_calls)) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update covariance
            y_c = Cinv_sqrt.T @ (m_new - m) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_c
            delta_hs = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (budget - n_calls))) ** 2 - (1 + 2 / (n + 1)) * chi_n
            C = (1 + cov_update * (cmu * delta_hs - 1)) * C \
                + c1 * (np.outer(p_c, p_c) + (1 - cc) * (1 - cc) * C) \
                + 2 * cmu * np.sum(w[:, None] * np.outer(ys_sorted[:mu] - m, (ys_sorted[:mu] - m).T), axis=0) / sigma ** 2
            
            # Symmetrize
            C = 0.5 * (C + C.T)
            
            # Update eigendecomposition
            update_eig()
            
            m = m_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
