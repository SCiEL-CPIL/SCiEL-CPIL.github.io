# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with shared low-rank factorization
    and dual-timescale Kalman filtering with adaptive process noise.
    
    Arms share a latent factor space: w_a = V @ z_a where V is (d x k) global
    context-to-latent map and z_a is (k x 1) arm-specific latent vector.
    This enforces structural coherence across arms for better sample efficiency.
    
    Uses:
    - Joint state tracking with adaptive process noise (tanh-saturated EMA)
    - Joseph-form covariance update for numerical stability
    - Count-based exploration decay
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.t0 = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Choose latent dimension: min(d, n_arms, 4) for low-rank structure
        self.k = max(1, min(dim, n_arms, 4))

        # Shared factor matrix V: (d x k)
        self.V = np.random.randn(dim, self.k) * 0.1
        # Arm-specific latent vectors z_a: (k,)
        self.z = [np.zeros(self.k) for _ in range(n_arms)]

        # Covariance for V (d x k x d x k) -- use Kronecker product structure
        # For tractability, maintain separate covariances:
        # P_V: (d x k) covariance in vectorized form -> use (d*k x d*k)
        # P_z[a]: (k x k) for each arm
        dk = self.d * self.k
        self.P_V = np.eye(dk) * 5.0
        self.P_z = [np.eye(self.k) * 5.0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.Q_V = 0.01  # base process noise for V
        self.Q_z = 0.01  # base process noise for z
        self.Q_V_ema = 0.01
        self.Q_z_ema = 0.01
        self.residual_ema = [0.0 for _ in range(n_arms)]

        # Count-based exploration
        self.count = [0 for _ in range(n_arms)]
        self.exploration_base = 1.0
        self.exploration_decay = 0.999

        # Track reward history for adaptive noise
        self.reward_history = []
        self.prev_rewards = [0.0 for _ in range(n_arms)]

    def _get_theta(self, a):
        """Get arm weight vector from factorization."""
        V_flat = self.V  # (d, k)
        return V_flat @ self.z[a]  # (d,)

    def _predict(self):
        """Kalman prediction step with adaptive process noise."""
        # Update process noise based on residual history
        if self.t > 10:
            # tanh-saturated adaptive Q
            mean_res = np.mean([self.residual_ema[a] for a in range(self.n)])
            self.Q_V_ema = 0.995 * self.Q_V_ema + 0.005 * np.tanh(mean_res) * 0.05
            self.Q_z_ema = 0.995 * self.Q_z_ema + 0.005 * np.tanh(mean_res) * 0.05
            self.Q_V = max(0.001, self.Q_V_ema)
            self.Q_z = max(0.001, self.Q_z_ema)

        # Predict: add process noise to covariances
        dk = self.d * self.k
        self.P_V += np.eye(dk) * self.Q_V
        for a in range(self.n):
            self.P_z[a] += np.eye(self.k) * self.Q_z

    def _update_V(self, context, arm, reward):
        """
        Update V given observation. The observation model:
        reward = (V @ z_arm)^T context + noise = sum_k z_arm[k] * (V[:,k]^T context)
        
        This is linear in V if z_arm is fixed.
        Let h = V @ z_arm (d,), then reward = h^T context.
        h = V z_arm, so dh/dV involves z_arm.
        
        For simplicity, treat the effective weight w = V @ z_arm and update
        both V and z jointly via a linearized approach.
        
        Alternative: fix z temporarily, update V as if it were a linear bandit
        with "context" being context * z_arm^T reshaped.
        """
        # Effective context for V: each element V[i,j] contributes z_arm[j] * context[i]
        # So the "context" for V (flattened) is: outer(context, z_arm) flattened
        z_a = self.z[arm]
        if np.linalg.norm(z_a) < 1e-6:
            # If z is near zero, update z instead
            self._update_z(context, arm, reward)
            return

        # Context for V: (d*k,) vector
        ctx_V = np.outer(context, z_a).flatten()  # (d*k,)

        # Predicted reward from V: (V z_arm)^T context = (V^T z_arm)^T context
        # = sum_i,j V[i,j] * z_arm[j] * context[i] = ctx_V^T V_flat
        V_flat = self.V.flatten()
        pred = ctx_V @ V_flat

        # Innovation
        innovation = reward - pred
        self.residual_ema[arm] = 0.99 * self.residual_ema[arm] + 0.01 * innovation**2

        # Kalman gain for V
        # P_V is (d*k x d*k), ctx_V is (d*k,)
        P_ctx = self.P_V @ ctx_V  # (d*k,)
        S = ctx_V @ P_ctx + 1.0  # scalar (observation noise = 1)
        K = P_ctx / S  # (d*k,)

        # Update V
        self.V_flat_new = V_flat + K * innovation
        self.V = self.V_flat_new.reshape(self.d, self.k)

        # Joseph-form covariance update
        I_K_ctx = np.eye(self.d * self.k) - np.outer(K, ctx_V)
        self.P_V = I_K_ctx @ self.P_V @ I_K_ctx.T + np.outer(K, K) * 1.0
        # Symmetrize for numerical stability
        self.P_V = (self.P_V + self.P_V.T) / 2.0

    def _update_z(self, context, arm, reward):
        """Update z_arm given observation, treating V as fixed."""
        V = self.V  # (d, k)
        # Effective context for z: V^T context (k,)
        ctx_z = V.T @ context  # (k,)

        # Predicted reward: (V z_arm)^T context = z_arm^T (V^T context) = z_arm^T ctx_z
        pred = self.z[arm] @ ctx_z

        innovation = reward - pred
        self.residual_ema[arm] = 0.99 * self.residual_ema[arm] + 0.01 * innovation**2

        # Kalman gain for z
        P_ctx = self.P_z[arm] @ ctx_z  # (k,)
        S = ctx_z @ P_ctx + 1.0
        K = P_ctx / S  # (k,)

        self.z[arm] = self.z[arm] + K * innovation

        # Joseph-form covariance update
        I_K_ctx = np.eye(self.k) - np.outer(K, ctx_z)
        self.P_z[arm] = I_K_ctx @ self.P_z[arm] @ I_K_ctx.T + np.outer(K, K) * 1.0
        self.P_z[arm] = (self.P_z[arm] + self.P_z[arm].T) / 2.0

    def select(self, context):
        """Select arm using UCB with count-based exploration decay."""
        # Check deadline
        if time.time() - self.t0 > self.deadline:
            # Return best arm based on current estimates
            scores = [self._get_theta(a) @ context for a in range(self.n)]
            return int(np.argmax(scores))

        self._predict()

        scores = []
        for a in range(self.n):
            theta = self._get_theta(a)  # (d,)
            mean = theta @ context

            # Uncertainty: sqrt(context^T (V P_z V^T) context)
            # P_z[a] is (k x k), so effective covariance for theta is V P_z[a] V^T (d x d)
            # But that's expensive. Use: sqrt(z_a^T P_z[a] z_a * ||V^T context||^2)
            # More precisely: Var(reward) = ctx_z^T P_z[a] ctx_z where ctx_z = V^T context
            ctx_z = self.V.T @ context  # (k,)
            var = ctx_z @ self.P_z[a] @ ctx_z
            uncertainty = np.sqrt(max(0.0, var))

            # Count-based exploration decay
            count_factor = self.exploration_base * (self.exploration_decay ** self.count[a])
            bonus = 1.0 * uncertainty * count_factor + 0.1 * count_factor / (1 + self.count[a])

            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model with observed reward."""
        self.t += 1
        self.count[arm] += 1
        self.reward_history.append(reward)

        # Update both V and z
        self._update_V(context, arm, reward)
        self._update_z(context, arm, reward)
# EVOLVE-BLOCK-END
