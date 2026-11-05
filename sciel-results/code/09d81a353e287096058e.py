# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 115.0
        
        # KRLS Parameters
        self.max_centers = 20  # Size of dictionary per arm
        self.gamma = 1.0       # RBF kernel width
        self.alpha = 0.1       # Regularization
        self.beta = 1.5        # Exploration scale
        self.lambda_min = 0.8
        self.lambda_max = 0.99
        self.err_alpha = 0.5   # Error sensitivity for adaptive lambda
        
        # State per arm
        self.centers = [[] for _ in range(n_arms)]
        self.weights = [np.array([]) for _ in range(n_arms)]
        self.P = [np.array([]) for _ in range(n_arms)]
        self.lambda_val = [0.95] * n_arms
        self.count = [0] * n_arms
        self.step = 0

    def _check_deadline(self):
        return (time.time() - self.t0) > self.deadline

    def _normalize(self, x):
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            return x / norm
        return x

    def _kernel(self, x, centers_arr):
        # x: (d,), centers_arr: (M, d)
        # Returns (M,)
        if centers_arr.size == 0:
            return np.array([])
        diff = centers_arr - x
        sq_dist = np.sum(diff**2, axis=1)
        return np.exp(-self.gamma * sq_dist)

    def _update_P(self, a):
        """Recompute P from scratch if dictionary changes, or just ensure it's valid."""
        M = len(self.centers[a])
        if M == 0:
            self.P[a] = np.array([])
            return
        centers_arr = np.array(self.centers[a])
        # Compute full kernel matrix K
        # K[i, j] = K(centers[i], centers[j])
        diff = centers_arr[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        K = np.exp(-self.gamma * sq_dist)
        # P = (K + alpha I)^-1
        reg = K + self.alpha * np.eye(M)
        try:
            self.P[a] = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            self.P[a] = np.linalg.pinv(reg)

    def select(self, context):
        ctx = np.asarray(context, dtype=float)
        ctx_norm = self._normalize(ctx)
        
        if self._check_deadline():
            scores = []
            for a in range(self.n):
                mean, _ = self._predict(a, ctx_norm)
                scores.append(mean)
            return int(np.argmax(scores))
            
        self.step += 1
        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            mean, var = self._predict(a, ctx_norm)
            # UCB
            # Add bonus for unvisited arms
            bonus = self.beta * np.sqrt(max(var, 0.0))
            if self.count[a] == 0:
                bonus += 2.0
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)
        ctx_norm = self._normalize(ctx)
        
        if self._check_deadline():
            return
            
        self._update_krls(arm, ctx_norm, float(reward))
        self.count[arm] += 1

    def _predict(self, a, ctx_norm):
        centers = self.centers[a]
        if len(centers) == 0:
            return 0.0, 1.0
        
        centers_arr = np.array(centers)
        k = self._kernel(ctx_norm, centers_arr)
        w = self.weights[a]
        P = self.P[a]
        
        mean = float(k @ w)
        # Variance: K(x,x) - k^T P k
        # K(x,x) = 1.0
        var = 1.0 - float(k @ P @ k)
        return mean, var

    def _update_krls(self, a, ctx_norm, reward):
        centers = self.centers[a]
        
        # 1. Handle Dictionary Growth
        if len(centers) < self.max_centers:
            # Check if context is sufficiently different from existing centers
            # If list is empty, just add.
            # If not, check min distance (or max kernel).
            if len(centers) > 0:
                centers_arr = np.array(centers)
                k_existing = self._kernel(ctx_norm, centers_arr)
                # If max kernel < 0.8, it's a new region
                if np.max(k_existing) < 0.8:
                    self.centers[a].append(ctx_norm.copy())
                    # Recompute P for new size
                    self._update_P(a)
                    # Initialize weight for new center to 0? 
                    # RLS will update it.
                    # But P is inverse of (K + alpha I).
                    # We need to extend weights.
                    self.weights[a] = np.append(self.weights[a], 0.0)
                    return
            else:
                self.centers[a].append(ctx_norm.copy())
                self._update_P(a)
                self.weights[a] = np.array([0.0])
                return

        # 2. RLS Update with Forgetting Factor
        # Adaptive Lambda
        # Estimate error from previous step? 
        # We can use the innovation (reward - pred)
        pred, _ = self._predict(a, ctx_norm)
        innov = reward - pred
        
        # Map error to lambda
        # Large error -> smaller lambda (more forgetting)
        # err_alpha controls sensitivity
        err_score = np.exp(-self.err_alpha * abs(innov))
        self.lambda_val[a] = self.lambda_min + (self.lambda_max - self.lambda_min) * err_score
        lam = self.lambda_val[a]
        
        centers_arr = np.array(self.centers[a])
        k = self._kernel(ctx_norm, centers_arr)
        w = self.weights[a]
        P = self.P[a]
        
        # RLS Update
        # K_gain = P k / (lam + k^T P k)
        denom = lam + float(k @ P @ k)
        K_gain = P @ k / denom
        
        # w_new = w + K_gain * innov
        w = w + K_gain * innov
        
        # P_new = (1/lam) * (P - K_gain k^T P)
        P = (1.0 / lam) * (P - np.outer(K_gain, k @ P))
        
        # Symmetrize P for numerical stability
        P = 0.5 * (P + P.T)
        
        self.weights[a] = w
        self.P[a] = P

# EVOLVE-BLOCK-END
