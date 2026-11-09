"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor via Cholesky and EWMA non-stationarity detection."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Initialize A and b for each arm
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Cholesky factors for numerical stability
        self.L = [np.eye(dim) for _ in range(n_arms)]
        self.L_valid = [True] * n_arms

        # Adaptive forgetting: base and EWMA-based adjustment
        self.base_gamma = 0.98
        self.gamma = self.base_gamma
        self.reward_ewma = [0.0] * n_arms
        self.reward_var_ewma = [0.0] * n_arms
        self.prev_reward = [0.0] * n_arms
        self.ewma_alpha = 0.1  # Smoothing for non-stationarity detection

        # Exploration coefficient
        self.alpha = 1.0

        # Track per-arm reward history for variance estimation
        self.arm_reward_history = [[] for _ in range(n_arms)]

    def _update_cholesky(self, arm):
        """Update Cholesky factor for arm's A matrix."""
        try:
            # Add outer product to A
            self.A[arm] = self.A[arm] + np.eye(self.d) * 0.1  # Small regularization
            # Recompute Cholesky (for small dim this is fast)
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            self.L_valid[arm] = True
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue-based if Cholesky fails
            self.L_valid[arm] = False
            # Add more regularization
            self.A[arm] += np.eye(self.d) * 1.0
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                self.L_valid[arm] = True
            except np.linalg.LinAlgError:
                pass

    def _solve_cholesky(self, arm, rhs):
        """Solve L L^T x = rhs using Cholesky factor."""
        if self.L_valid[arm]:
            # Forward substitution: L y = rhs
            y = np.linalg.solve(self.L[arm], rhs)
            # Back substitution: L^T x = y
            x = np.linalg.solve(self.L[arm].T, y)
            return x
        else:
            # Fallback to direct solve
            return np.linalg.solve(self.A[arm], rhs)

    def _variance_cholesky(self, arm, context):
        """Compute context^T A^{-1} context using Cholesky factor."""
        if self.L_valid[arm]:
            # Solve L z = context, then variance = z^T z
            z = np.linalg.solve(self.L[arm], context)
            return np.dot(z, z)
        else:
            # Fallback
            inv_A = np.linalg.inv(self.A[arm])
            return context @ inv_A @ context

    def _adapt_forgetting(self, arm, reward):
        """Adapt forgetting factor based on non-stationarity detection."""
        # Update EWMA of reward
        if self.t > 0:
            self.reward_ewma[arm] = (1 - self.ewma_alpha) * self.reward_ewma[arm] + self.ewma_alpha * reward
            # Update variance EWMA
            diff = reward - self.prev_reward[arm]
            self.reward_var_ewma[arm] = (1 - self.ewma_alpha) * self.reward_var_ewma[arm] + self.ewma_alpha * (diff ** 2)
        else:
            self.reward_ewma[arm] = reward
            self.reward_var_ewma[arm] = 0.0

        self.prev_reward[arm] = reward

        # Estimate non-stationarity: high variance in reward changes suggests drift
        # Normalize variance to get a measure of change
        if self.reward_var_ewma[arm] > 1e-8:
            # Higher variance -> more non-stationary -> smaller gamma (faster forgetting)
            # Map variance to gamma range [0.9, 0.999]
            norm_var = min(self.reward_var_ewma[arm] / (self.reward_var_ewma[arm] + 1.0), 1.0)
            self.gamma = 0.999 - 0.099 * norm_var
        else:
            self.gamma = 0.999

        # Clamp gamma
        self.gamma = max(0.9, min(0.999, self.gamma))

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return the arm with highest estimated mean reward
            scores = []
            for a in range(self.n):
                try:
                    theta = self._solve_cholesky(a, self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            try:
                # Get theta = A^{-1} b
                theta = self._solve_cholesky(a, self.b[a])
                mean = theta @ context

                # Get variance for UCB
                var = self._variance_cholesky(a, context)
                ucb = mean + self.alpha * np.sqrt(var)
                scores.append(ucb)
            except:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Adapt forgetting factor
        self._adapt_forgetting(arm, reward)

        # Apply forgetting
        gamma = self.gamma
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Update Cholesky factor
        self._update_cholesky(arm)

        self.t += 1
# EVOLVE-BLOCK-END
