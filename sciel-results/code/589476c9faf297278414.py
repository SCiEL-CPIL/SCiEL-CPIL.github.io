"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Kernelized non-stationary linear bandit using time-dependent RBF feature mapping."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

        # RBF kernel parameters
        self.gamma = 0.5  # RBF width parameter
        self.n_landmarks = 8  # Number of RBF landmarks per arm

        # Per-arm state: A (regularized Gram matrix), b (reward vector), landmarks
        self.A = [np.eye(self.n_landmarks) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(self.n_landmarks) for _ in range(n_arms)]
        self.landmarks = [np.random.randn(self.n_landmarks, dim) * 0.5 for _ in range(n_arms)]
        self.landmark_times = [np.zeros(self.n_landmarks) for _ in range(n_arms)]

        # Precomputed inverses
        self.inv_A = [np.linalg.inv(self.A[a]) for a in range(n_arms)]

        # Track recent contexts for landmark updates
        self.context_history = [[] for _ in range(n_arms)]
        self.max_history = 50

    def _rbf_kernel(self, x, y, t_x, t_y):
        """Time-dependent RBF kernel: K(x, t_x; y, t_y) = exp(-gamma * ||x-y||^2) * exp(-gamma * |t_x - t_y|^2)"""
        spatial_dist = np.sum((x - y) ** 2)
        temporal_dist = (t_x - t_y) ** 2
        return np.exp(-self.gamma * spatial_dist - self.gamma * temporal_dist)

    def _compute_kernel_vector(self, context, arm):
        """Compute kernel values between context and all landmarks for given arm."""
        k = np.zeros(self.n_landmarks)
        for i in range(self.n_landmarks):
            k[i] = self._rbf_kernel(context, self.landmarks[arm][i], self.t, self.landmark_times[arm][i])
        return k

    def _select_landmark_to_update(self, arm):
        """Select which landmark to update based on recency and relevance."""
        # Prefer updating the oldest landmark
        oldest_idx = np.argmin(self.landmark_times[arm])
        return oldest_idx

    def select(self, context):
        if time.time() > self.deadline:
            return 0

        self.t += 1
        scores = []
        for a in range(self.n):
            k = self._compute_kernel_vector(context, a)
            # Prediction: theta^T * k where theta = A^{-1} b
            pred = k @ self.inv_A[a] @ self.b[a]
            # Variance: k^T A^{-1} k
            var = k @ self.inv_A[a] @ k
            ucb = pred + 0.5 * np.sqrt(max(var, 0))
            scores.append(ucb)

        best_arm = int(np.argmax(scores))
        self._last_arm = best_arm
        return best_arm

    def update(self, context, arm, reward):
        # Update the oldest landmark for this arm
        idx = self._select_landmark_to_update(arm)
        self.landmarks[arm][idx] = context.copy()
        self.landmark_times[arm][idx] = self.t

        # Store context in history
        self.context_history[arm].append((context.copy(), self.t))
        if len(self.context_history[arm]) > self.max_history:
            self.context_history[arm].pop(0)

        # Compute kernel vector for the new context
        k = self._compute_kernel_vector(context, arm)

        # Update A and b (kernel ridge regression)
        self.A[arm] = self.A[arm] + np.outer(k, k)
        self.b[arm] = self.b[arm] + reward * k

        # Recompute inverse using Sherman-Morrison formula for efficiency
        # A_new = A_old + k*k^T
        # A_new^{-1} = A_old^{-1} - (A_old^{-1} k k^T A_old^{-1}) / (1 + k^T A_old^{-1} k)
        Ak = self.inv_A[arm] @ k
        denom = 1.0 + k @ Ak
        if abs(denom) > 1e-10:
            self.inv_A[arm] = self.inv_A[arm] - np.outer(Ak, Ak) / denom
        else:
            # Fallback to direct inversion if Sherman-Morrison fails
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
# EVOLVE-BLOCK-END
