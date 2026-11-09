"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """
    Bayesian Optimization for BBOB f16 (Weierstrass) using GP surrogate
    with Expected Improvement acquisition.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf
        self._X = []  # observed points
        self._Y = []  # observed values

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        self._X.append(np.array(x, dtype=np.float64))
        self._Y.append(float(fx))
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = np.array(x, dtype=np.float64)

    def _gp_predict(self, X_new, X_train, Y_train, length_scale=1.0, sigma_f=1.0, noise=1e-8):
        """
        Gaussian Process prediction.
        Returns mean and std for X_new.
        """
        n = len(X_train)
        if n == 0:
            return np.zeros(len(X_new)), np.full(len(X_new), sigma_f)
        
        # RBF kernel
        def kernel(A, B):
            sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (length_scale**2))
        
        K = kernel(X_train, X_train) + noise * np.eye(n)
        K_star = kernel(X_train, X_new)
        
        try:
            alpha = np.linalg.solve(K, Y_train)
            mean = K_star.T @ alpha
            v = np.linalg.solve(K, K_star)
            var = sigma_f**2 - np.sum(K_star**2 * v, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            mean = np.full(len(X_new), np.mean(Y_train))
            std = np.full(len(X_new), np.std(Y_train) + 1e-6)
        
        return mean, std

    def _expected_improvement(self, mean, std, best_f, xi=0.01):
        """Compute Expected Improvement."""
        improvement = best_f - mean - xi
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std > 1e-10, ei, 0)
        return ei

    def _acquire_next_point(self, rng, n_candidates=200):
        """
        Select next point using EI acquisition with random candidates.
        """
        n = self.dim
        X_train = np.array(self._X)
        Y_train = np.array(self._Y)
        
        # Adaptive length scale based on spread of data
        if len(self._X) >= 2:
            spread = np.std(X_train, axis=0)
            length_scale = np.clip(np.mean(spread), 0.1, 5.0)
        else:
            length_scale = 1.0
        
        # Adaptive sigma_f
        if len(self._Y) >= 2:
            sigma_f = np.clip(np.std(Y_train), 0.1, 10.0)
        else:
            sigma_f = 1.0
        
        # Generate random candidates
        candidates = rng.uniform(-5, 5, (n_candidates, n))
        
        # Predict and compute EI
        mean, std = self._gp_predict(candidates, X_train, Y_train, length_scale, sigma_f)
        ei = self._expected_improvement(mean, std, self._best_f)
        
        # Also consider points near current best with small perturbations
        if self._best_x is not None and len(self._X) > 5:
            local_candidates = self._best_x[None, :] + rng.normal(0, 0.1, (50, n))
            local_candidates = np.clip(local_candidates, -5, 5)
            mean_local, std_local = self._gp_predict(local_candidates, X_train, Y_train, length_scale, sigma_f)
            ei_local = self._expected_improvement(mean_local, std_local, self._best_f)
            
            # Combine: pick from global if EI is high, otherwise from local
            global_best_idx = np.argmax(ei)
            local_best_idx = np.argmax(ei_local)
            
            # 70% chance global, 30% chance local
            if rng.random() < 0.7:
                return candidates[global_best_idx]
            else:
                return local_candidates[local_best_idx]
        
        return candidates[np.argmax(ei)]

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 295.0  # safety margin before 330s kill
        
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial random sampling (20% of budget)
        n_init = max(10, self.budget // 5)
        for i in range(n_init):
            if self._eval_count >= self.budget:
                break
            if (time.time() - self._start_time) > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = float(func(x))
            self._record(x, f)
        
        # Phase 2: Bayesian Optimization with GP + EI
        while self._eval_count < self.budget:
            if (time.time() - self._start_time) > deadline:
                break
            
            # Check if we have enough data for GP
            if len(self._X) < 2:
                x = rng.uniform(-5, 5, n)
                f = float(func(x))
                self._record(x, f)
                continue
            
            try:
                x_next = self._acquire_next_point(rng)
                f_next = float(func(x_next))
                self._record(x_next, f_next)
            except Exception:
                # Fallback to random sampling
                x = rng.uniform(-5, 5, n)
                f = float(func(x))
                self._record(x, f)
        
        # Phase 3: Local polish with coordinate descent if budget remains
        remaining = self.budget - self._eval_count
        if remaining > 5 and self._best_x is not None:
            x = self._best_x.copy()
            f = self._best_f
            step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]
            
            for ss in step_sizes:
                if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
                    break
                improved = True
                while improved and self._eval_count < self.budget:
                    improved = False
                    for i in range(n):
                        if (time.time() - self._start_time) > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            self._record(x_trial, f_trial)
                            if f_trial < f - 1e-12:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                        if self._eval_count >= self.budget:
                            break
                    if self._eval_count >= self.budget:
                        break
                
                if f < self._best_f:
                    self._best_f = f
                    self._best_x = x.copy()
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
