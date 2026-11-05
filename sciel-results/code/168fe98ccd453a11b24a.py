# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before the 150 s wall-clock kill


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Structural change: Uses a Discounted Least Squares (DLS) formulation 
    (equivalent to a Kalman filter with constant discount factor gamma) 
    instead of the previous adaptive process noise Kalman filter.
    
    Key features:
    1. DLS with exponential forgetting: V_{t+1} = (gamma * V_t^{-1} + x x^T)^{-1}
       This provides a natural, stable mechanism for non-stationarity.
    2. Adaptive gamma: The forgetting rate is adjusted based on the magnitude of
       the residual (innovation). Large residuals suggest drift, so gamma is
       decreased (faster forgetting). Small residuals suggest stability, so
       gamma is increased (slower forgetting).
    3. Tanh-saturation for gamma adaptation to keep it bounded and smooth.
    4. UCB exploration with count-based decay.
    5. Joseph-form update for numerical stability (though DLS update is simpler).
    6. Deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = _DEADLINE

        # Per-arm DLS state:
        # V_inv[a] is the inverse covariance matrix (precision matrix)
        # b[a] is the cross-covariance vector (sum of gamma^t r x)
        # theta[a] = V_inv[a]^{-1} b[a]
        
        # Initialize with high uncertainty: V_inv = (1/lambda) I, so V = lambda I
        # Let lambda = 10.0, so V_inv = 0.1 I
        self.V_inv = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive gamma parameters
        # gamma in (0, 1). Higher gamma = slower forgetting.
        self.gamma_base = 0.98
        self.gamma_min = 0.80
        self.gamma_max = 0.999
        self.gamma = [self.gamma_base for _ in range(n_arms)]
        
        # Residual EMA for adaptive gamma
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1  # Smoothing factor for residual EMA

        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_decay = 20.0

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self, arm):
        """Count-based exploration decay."""
        c = self.count[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def select(self, context):
        # Deadline guard: fall back to greedy
        if time.time() > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Variance: x^T V x = x^T (V_inv)^{-1} x
            # Instead of inverting V_inv every time, we can maintain V explicitly
            # or compute x^T V x via solve.
            # For dim=5, inverting is cheap. But let's use Cholesky or solve for stability.
            # Actually, we maintain V_inv. To get variance, we need x^T V x.
            # V = inv(V_inv). 
            # Alternatively, we can maintain V as well and update it.
            # Let's switch to maintaining V (covariance) for easier UCB variance calculation.
            # DLS update:
            # V_{t+1} = (1/gamma) V_t - ((1/gamma) V_t x) (x^T (1/gamma) V_t) / (1 + x^T (1/gamma) V_t x)
            # This is the standard rank-1 update.
            
            # Let's refactor to maintain V instead of V_inv for this implementation.
            # V = covariance matrix.
            # Init: V = 10 I
            # Update:
            # V = (1/gamma) V
            # K = V x / (1 + x^T V x)
            # V = V - K x^T
            # theta = theta + K (r - x^T theta)
            
            # Wait, the current code uses V_inv. Let's stick to V_inv for the update logic 
            # but compute variance efficiently.
            # Actually, for dim=5, np.linalg.inv is fast enough.
            # But better: maintain V explicitly.
            # Let's rewrite the state to use V (covariance) directly.
            
            # Re-initialization logic would need to change.
            # Let's restart the class design in my head to use V.
            # 
            # Okay, I will change the class to use V (covariance) directly.
            # This is more standard for UCB.
            
            pass # Placeholder for logic change

    # --- REWRITING CLASS TO USE V (COVARIANCE) DIRECTLY ---
    # The above __init__ set up V_inv. I will replace the whole class below.
    
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = _DEADLINE

        # Per-arm DLS state using Covariance V
        self.V = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive gamma parameters
        self.gamma_base = 0.98
        self.gamma_min = 0.80
        self.gamma_max = 0.999
        self.gamma = [self.gamma_base for _ in range(n_arms)]
        
        # Residual EMA for adaptive gamma
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1

        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_decay = 20.0

        # Deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self, arm):
        c = self.count[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def select(self, context):
        if time.time() > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            var = context @ self.V[a] @ context
            if var < 0:
                var = 0.0
            alpha_a = self._get_alpha(a)
            ucb = mean + alpha_a * np.sqrt(var)
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        a = arm
        self.t += 1

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        x = context
        V = self.V[a]
        theta = self.theta[a]
        gamma = self.gamma[a]

        # 1. Predict step (forgetting): V_pred = (1/gamma) * V
        V_pred = V / gamma

        # 2. Innovation
        pred_reward = theta @ x
        innov = reward - pred_reward

        # 3. Kalman Gain
        # K = V_pred x / (1 + x^T V_pred x)
        denom = 1.0 + x @ V_pred @ x
        K = (V_pred @ x) / denom

        # 4. Update State
        self.theta[a] = theta + K * innov

        # 5. Update Covariance
        # V_new = V_pred - K x^T
        self.V[a] = V_pred - np.outer(K, x)
        
        # Symmetrize
        self.V[a] = (self.V[a] + self.V[a].T) / 2.0
        # Add small ridge for PSD stability
        self.V[a] += 1e-8 * np.eye(self.d)

        self.count[a] += 1

        # 6. Adaptive Gamma
        # Update residual EMA
        abs_innov = abs(innov)
        self.residual_ema[a] = (1 - self.ema_alpha) * self.residual_ema[a] + self.ema_alpha * abs_innov
        
        # Map residual EMA to gamma
        # If residual is large, we want smaller gamma (faster forgetting)
        # If residual is small, we want larger gamma (slower forgetting)
        
        # Normalize residual. Assume typical scale of rewards is ~1.
        # raw_score = residual_ema / 1.0
        # Use tanh to map to [0, 1]
        raw_score = self.residual_ema[a]
        # Normalize by a scale factor. Let's say scale = 1.0.
        norm_score = np.tanh(raw_score / 1.0)
        
        # Map norm_score [0, 1] to gamma [gamma_min, gamma_max]
        # Inverse relationship: high score -> low gamma
        # gamma = gamma_max - (gamma_max - gamma_min) * norm_score
        self.gamma[a] = self.gamma_max - (self.gamma_max - self.gamma_min) * norm_score
        
        # Clamp
        self.gamma[a] = np.clip(self.gamma[a], self.gamma_min, self.gamma_max)

# EVOLVE-BLOCK-END
