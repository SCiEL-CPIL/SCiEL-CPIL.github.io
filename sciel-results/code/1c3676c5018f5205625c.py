# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: GP surrogate with EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hv_2d(self, F, ref):
        """Exact 2D HV for points strictly below ref. F: (n,2)."""
        n = len(F)
        if n == 0:
            return 0.0
        order = np.argsort(F[:, 0])
        F = F[order]
        hv = 0.0
        max_y = -np.inf
        for i in range(n):
            dx = ref[0] - F[i, 0]
            if dx <= 0:
                break
            dy = ref[1] - F[i, 1]
            if dy > max_y:
                max_y = dy
                if max_y > 0:
                    hv += dx * max_y
        return hv

    def _hv_3d(self, F, ref):
        """Exact 3D HV for points strictly below ref. F: (n,3)."""
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]

        # Keep only the Pareto front (points not dominated by an earlier point).
        keep = np.ones(n, dtype=bool)
        best_y = np.inf
        best_z = np.inf
        for i in range(n):
            y = Fv[i, 1]
            z = Fv[i, 2]
            if y <= best_y and z <= best_z and (y < best_y or z < best_z):
                best_y, best_z = y, z
            else:
                keep[i] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sweep along objective 0.
        hv = 0.0
        prev_x = -np.inf
        prev = Fv[0]
        for i in range(1, n):
            cur = Fv[i]
            dx = cur[0] - prev_x
            if dx > 0:
                hv += dx * self._hv_2d(prev.reshape(1, 2), ref[1:])
            prev = cur
            prev_x = cur[0]
        dx = ref[0] - prev_x
        if dx > 0:
            hv += dx * self._hv_2d(prev.reshape(1, 2), ref[1:])
        return hv

    def _gp_predict(self, X_train, y_train, X_test, length_scale, noise):
        """GP prediction with RBF kernel. Returns mean, std."""
        from scipy.spatial.distance import cdist
        n = X_train.shape[0]
        m = X_test.shape[0]
        # Normalize inputs
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-10
        Xtr = (X_train - mu) / sigma
        Xte = (X_test - mu) / sigma
        # RBF kernel
        d = cdist(Xtr, Xtr)
        K = np.exp(-0.5 * (d / length_scale) ** 2) + noise * np.eye(n)
        d_test = cdist(Xte, Xtr)
        K_star = np.exp(-0.5 * (d_test / length_scale) ** 2)
        # Solve
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            K_reg = K + 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K_reg)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def _ehvi_acquisition(self, X_train, F_train, X_cand, ref, length_scale, noise, n_samples=200):
        """Compute EHVI acquisition for candidate points."""
        n_cand = X_cand.shape[0]
        # Filter valid points (strictly below ref)
        valid = np.all(F_train < ref, axis=1)
        F_valid = F_train[valid]
        X_valid = X_train[valid]
        
        if len(F_valid) == 0:
            # No valid points, use exploration
            return np.ones(n_cand) * 1e-6
        
        # Current HV
        hv_current = self._hv_3d(F_valid, ref)
        
        # For each candidate, predict all 3 objectives
        means = np.zeros((n_cand, self.n_obj))
        stds = np.zeros((n_cand, self.n_obj))
        for j in range(self.n_obj):
            m, s = self._gp_predict(X_valid, F_valid[:, j], X_cand, length_scale, noise)
            means[:, j] = m
            stds[:, j] = s
        
        # MC estimate of EHVI
        rng = np.random.default_rng(42)
        ehvi = np.zeros(n_cand)
        
        for i in range(n_cand):
            # Sample from posterior
            F_sample = means[i] + stds[i] * rng.standard_normal((n_samples, self.n_obj))
            # Clip to be reasonable
            F_sample = np.clip(F_sample, 0, None)
            
            # For each sample, compute HV with this point added
            hv_samples = 0.0
            for s in range(n_samples):
                F_new = np.vstack([F_valid, F_sample[s:s+1]])
                hv_samples += self._hv_3d(F_new, ref)
            hv_samples /= n_samples
            ehvi[i] = max(0, hv_samples - hv_current)
        
        return ehvi

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        X_list = []
        F_list = []

        def evaluate(x):
            f = np.asarray(problem(x), dtype=float)
            if f.shape != (self.n_obj,):
                f = f.reshape(-1)[:self.n_obj]
            X_list.append(np.asarray(x, dtype=float).copy())
            F_list.append(f)
            return f

        # ---- Initial random sampling (Latin Hypercube for better coverage) ----
        n_init = min(30, self.budget // 4)
        for i in range(n_init):
            if time.time() > deadline:
                break
            # LHS sampling
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + span[d] * (i + rng.random()) / n_init
            evaluate(x)

        used = len(X_list)

        # GP hyperparameters
        length_scale = 0.5
        noise = 1e-4
        n_cand = 200
        n_ehvi_samples = 100  # MC samples for EHVI

        # ---- Main BO loop ----
        while used < self.budget:
            if time.time() > deadline - 10.0:
                break

            F_arr = np.array(F_list)
            X_arr = np.array(X_list)
            
            # Filter valid points
            valid = np.all(F_arr < ref, axis=1)
            X_valid = X_arr[valid]
            F_valid = F_arr[valid]

            if len(X_valid) < 3:
                # Not enough data, do random exploration
                x = lo + span * rng.random(self.dim)
                evaluate(x)
                used += 1
                continue

            # Generate candidate points
            # Mix of random and perturbed existing points
            n_rand = n_cand // 2
            n_perturb = n_cand - n_rand
            
            X_cand = np.zeros((n_cand, self.dim))
            # Random candidates
            X_cand[:n_rand] = lo + span * rng.random((n_rand, self.dim))
            # Perturbed candidates around existing points
            for i in range(n_perturb):
                idx = int(rng.integers(len(X_valid)))
                X_cand[n_rand + i] = X_valid[idx] + 0.1 * span * rng.standard_normal(self.dim)
                X_cand[n_rand + i] = np.clip(X_cand[n_rand + i], lo, hi)

            # Compute EHVI acquisition
            ehvi = self._ehvi_acquisition(X_valid, F_valid, X_cand, ref, length_scale, noise, n_ehvi_samples)

            # Select best candidate
            best_idx = int(np.argmax(ehvi))
            x_next = X_cand[best_idx]
            x_next = np.clip(x_next, lo, hi)
            
            evaluate(x_next)
            used += 1

            # Occasionally adapt length_scale
            if used % 20 == 0:
                # Simple adaptation: if recent points are close, decrease LS
                if len(F_list) > 10:
                    recent = X_arr[-5:]
                    avg_dist = np.mean(np.linalg.norm(recent[:, None, :] - recent[None, :, :], axis=2))
                    if avg_dist < 0.2:
                        length_scale = max(0.2, length_scale * 0.9)
                    elif avg_dist > 0.8:
                        length_scale = min(1.5, length_scale * 1.1)

        return X_list
# EVOLVE-BLOCK-END
