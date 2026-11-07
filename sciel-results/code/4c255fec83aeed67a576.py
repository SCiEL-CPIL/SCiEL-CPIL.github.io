"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer with local refinement for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        def safe_func(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            total_calls += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Initial exploration with random sampling ---
        # Evaluate a set of diverse initial points
        rng = np.random.default_rng(42)
        
        # Center point and some random points
        initial_points = [np.zeros(n)]
        for _ in range(9):
            initial_points.append(rng.uniform(-5, 5, n))
        
        # Also add some points near the boundaries
        for _ in range(6):
            p = rng.uniform(-5, 5, n)
            # Push some coordinates toward boundaries
            mask = rng.random(n) < 0.3
            p[mask] = np.where(rng.random(np.sum(mask)) < 0.5, -5.0, 5.0)
            initial_points.append(p)
        
        # Evaluate initial points
        X = np.array(initial_points)
        Y = np.array([safe_func(x) for x in X])
        
        # Keep only finite values
        mask = np.isfinite(Y)
        X = X[mask]
        Y = Y[mask]
        
        if len(X) == 0:
            # Fallback: just do random search
            for _ in range(max_calls - total_calls):
                if not time_check():
                    break
                x = rng.uniform(-5, 5, n)
                safe_func(x)
            if best_x is None:
                best_x = np.zeros(n)
            return best_x, best_f
        
        # Sort by fitness
        idx = np.argsort(Y)
        X = X[idx]
        Y = Y[idx]
        
        # --- Phase 2: Surrogate-assisted search ---
        # Build a simple surrogate model (Gaussian Process with RBF kernel)
        
        def gp_predict(X_train, Y_train, X_test):
            """Simple GP prediction with RBF kernel."""
            # Standardize
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            Y_mean = Y_train.mean()
            Y_std = Y_train.std() + 1e-8
            
            X_train_s = (X_train - X_mean) / X_std
            X_test_s = (X_test - X_mean) / X_std
            Y_train_s = (Y_train - Y_mean) / Y_std
            
            # RBF kernel
            def rbf_kernel(A, B, length_scale=1.0):
                sq_dists = np.sum(A**2, axis=1)[:, None] - 2 * A @ B.T + np.sum(B**2, axis=1)[None, :]
                return np.exp(-0.5 * sq_dists / length_scale**2)
            
            # Choose length scale based on data
            length_scale = 1.0
            K = rbf_kernel(X_train_s, X_train_s, length_scale)
            K += 1e-6 * np.eye(len(X_train_s))  # noise for numerical stability
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_s))
            except np.linalg.LinAlgError:
                # Fallback to mean
                return np.full(X_test.shape[0], Y_mean), np.full(X_test.shape[0], Y_std)
            
            K_s = rbf_kernel(X_train_s, X_test_s, length_scale)
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = 1.0 - np.sum(v**2, axis=0)
            var = np.maximum(var, 0.0)
            std = np.sqrt(var) * Y_std
            
            mu = mu * Y_std + Y_mean
            std = std
            
            return mu, std
        
        def expected_improvement(mu, std, best_y, xi=0.01):
            """Expected improvement acquisition function."""
            imp = best_y - mu - xi
            z = imp / (std + 1e-8)
            ei = imp * norm.cdf(z) + std * norm.pdf(z)
            ei[std < 1e-8] = 0.0
            return ei
        
        # Surrogate-assisted loop
        # We'll use a mix of surrogate-guided and local search
        local_radius = 2.0
        n_surrogate_candidates = 20
        n_local_candidates = 5
        
        while total_calls < max_calls and time_check():
            # Get current best
            cur_best_idx = np.argmin(Y)
            cur_best_x = X[cur_best_idx]
            cur_best_f = Y[cur_best_idx]
            
            # Generate candidate points
            candidates = []
            
            # 1. Surrogate-guided candidates
            if len(X) >= 3:
                # Sample many random points and evaluate EI
                n_sample = 200
                X_sample = rng.uniform(-5, 5, (n_sample, n))
                try:
                    mu, std = gp_predict(X, Y, X_sample)
                    ei = expected_improvement(mu, std, cur_best_f, xi=0.01)
                    # Top EI candidates
                    top_idx = np.argsort(ei)[-n_surrogate_candidates:]
                    candidates.extend(X_sample[top_idx].tolist())
                except Exception:
                    pass
            
            # 2. Local search around best point
            for _ in range(n_local_candidates):
                x_local = cur_best_x + rng.normal(0, local_radius, n)
                x_local = np.clip(x_local, -5, 5)
                candidates.append(x_local)
            
            # 3. A few purely random points for exploration
            for _ in range(3):
                candidates.append(rng.uniform(-5, 5, n))
            
            # Deduplicate and evaluate
            evaluated = []
            for x in candidates:
                if total_calls >= max_calls or not time_check():
                    break
                x = np.clip(np.asarray(x, dtype=float), -5, 5)
                # Check if already evaluated (simple distance check)
                dists = np.linalg.norm(X - x, axis=1)
                if np.min(dists) < 1e-6:
                    continue
                f_val = safe_func(x)
                if np.isfinite(f_val):
                    evaluated.append((x, f_val))
            
            if not evaluated:
                # No new points evaluated, try smaller radius
                local_radius *= 0.5
                if local_radius < 0.01:
                    break
                continue
            
            # Update surrogate data
            new_X = np.array([e[0] for e in evaluated])
            new_Y = np.array([e[1] for e in evaluated])
            X = np.vstack([X, new_X])
            Y = np.concatenate([Y, new_Y])
            
            # Keep only the best 200 points to limit GP cost
            if len(X) > 200:
                idx = np.argsort(Y)[:200]
                X = X[idx]
                Y = Y[idx]
            
            # Adapt local radius based on recent improvements
            if Y.min() < cur_best_f:
                local_radius = max(local_radius * 0.8, 0.1)
            else:
                local_radius = min(local_radius * 1.2, 5.0)
        
        # --- Phase 3: Final local refinement around the best point ---
        if best_x is not None and total_calls < max_calls and time_check():
            # Do a simple coordinate descent / local search
            x_curr = best_x.copy()
            f_curr = best_f
            step = 1.0
            
            for _ in range(50):
                if total_calls >= max_calls or not time_check():
                    break
                
                improved = False
                for dim_idx in range(n):
                    for direction in [-1, 1]:
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if np.isfinite(f_trial) and f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break
                else:
                    step = min(step * 1.1, 2.0)
        
        if best_x is None:
            best_x = np.zeros(n)
            if total_calls < max_calls:
                best_f = safe_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
