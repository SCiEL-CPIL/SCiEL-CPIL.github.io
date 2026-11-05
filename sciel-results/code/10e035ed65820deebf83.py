"""Non-stationary linear contextual bandit with Robust Recursive Least Squares (RLS)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Robust RLS.
    
    Structural Change:
    - Replaces Kalman Filter with Recursive Least Squares (RLS).
    - Uses Huber-loss-inspired robust updates (clipping large innovations).
    - Adaptive forgetting factor: increases forgetting rate (lowers gamma) 
      when large residuals are detected (regime shift).
    - P-matrix inflation on outliers to maintain exploration/adaptability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 130.0  # Safety margin

        # Per-arm RLS state
        # theta: weight vector
        # P: inverse of the accumulated Hessian (covariance-like)
        # Initialize P with large value to allow quick initial learning
        init_P_val = 100.0
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * init_P_val for _ in range(n_arms)]
        
        # Exploration
        self.alpha = 1.2
        
        # RLS Parameters
        self.base_gamma = 0.95  # Forgetting factor (closer to 1 = slower forgetting)
        self.outlier_gamma = 0.8 # Faster forgetting on outliers
        self.regularization = 1e-4 # Small lambda for numerical stability
        self.huber_delta_factor = 2.0 # Threshold multiplier for std dev
        
        # Tracking for deadline
        self.best_arm = 0
        self.best_reward = -np.inf

    def select(self, context):
        if time.time() - self._t0 > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Variance estimate for UCB
            var = context @ self.P[a] @ context
            ucb = mean + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        a = arm
        self.t += 1

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # 1. Predict
        y_pred = self.theta[a] @ context
        y = reward - y_pred
        
        # 2. Compute Gain
        # S = x^T P x + lambda
        Px = self.P[a] @ context
        S = context @ Px + self.regularization
        
        # Gain K
        K = Px / S
        
        # 3. Robustness Check (Huber-like)
        # Estimate std of prediction error: sqrt(S)
        std_err = np.sqrt(S)
        threshold = self.huber_delta_factor * std_err
        
        is_outlier = abs(y) > threshold
        
        if is_outlier:
            # Clip the innovation to limit the impact of outliers on theta
            y_eff = np.sign(y) * threshold
            # Use faster forgetting (lower gamma) to adapt to regime change
            gamma = self.outlier_gamma
            # Inflate P to maintain high uncertainty/exploration
            # This counteracts the reduction from the RLS update
            inflation = 0.1 * np.eye(self.d)
        else:
            y_eff = y
            gamma = self.base_gamma
            inflation = None

        # 4. Update Theta
        self.theta[a] = self.theta[a] + K * y_eff
        
        # 5. Update P
        # Standard RLS: P_new = (1/gamma) * (I - K x^T) P_old
        # We compute (I - K x^T) P = P - K (x^T P) = P - K (S - lambda) approx P - K S?
        # Actually x^T P = (P x)^T = (K S)^T = K S (scalar)
        # So (I - K x^T) P = P - K (x^T P) = P - K (S - reg)
        # Let's just use the matrix form for clarity and stability, or the scalar shortcut.
        # Shortcut: P_new = (P - K (x^T P)) / gamma
        # x^T P = S - reg
        term = self.P[a] - np.outer(K, context) * (S - self.regularization)
        self.P[a] = term / gamma
        
        # Apply inflation if outlier
        if inflation is not None:
            self.P[a] = self.P[a] + inflation

        # Ensure symmetry
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        
        # Optional: Clip eigenvalues to prevent numerical issues (negative values)
        # This is expensive, so we only do it every few steps or if t is small
        if self.t % 10 == 0 or self.t < 50:
            eigvals, eigvecs = np.linalg.eigh(self.P[a])
            eigvals = np.maximum(eigvals, 1e-6)
            self.P[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
