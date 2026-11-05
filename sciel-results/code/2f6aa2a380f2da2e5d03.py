# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Exponentially-weighted linear bandit with Sherman-Morrison rank-1 updates 
    and innovation-adaptive forgetting.
    
    Combines the structural efficiency of independent per-arm linear models 
    with a robust adaptive forgetting mechanism based on the exponential 
    moving average of squared prediction errors (innovations).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A as identity (ridge) for each arm - stronger ridge for stability
        # 0.2 provides better conditioning than 0.1 without over-regularizing
        self.A = [np.eye(dim) * 0.2 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Precompute inverses for Sherman-Morrison updates
        self.A_inv = [np.eye(dim) / 0.2 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor (per-arm for independent adaptation)
        # Start aggressive (0.92) to quickly adapt to the initial environment
        self.gamma = [0.92] * n_arms
        # Exploration parameter
        self.alpha = 1.8
        
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit
        self.round_count = 0
        
        # Per-arm visit counts
        self.Visits = [0] * n_arms
        
        # Innovation tracking for adaptive gamma (per-arm)
        # Stores EMA of squared innovation (prediction error) for each arm
        self.innovation_sq = [0.0 for _ in range(n_arms)]

    def _sherman_morrison_update(self, arm, context, reward, gamma=None):
        if gamma is None:
            gamma = self.gamma[arm]
        """
        Update A_inv and theta using Sherman-Morrison formula 
        after a rank-1 update with forgetting factor gamma.
        
        A_new = gamma * A_old + x x^T
        b_new = gamma * b_old + reward * x
        """
        x = context
        inv = self.A_inv[arm]
        g = gamma
        
        # Sherman-Morrison: (A + u v^T)^-1 = A^-1 - A^-1 u v^T A^-1 / (1 + v^T A^-1 u)
        # Here, the update is effectively on the inverse of the decayed matrix.
        # We maintain A_inv such that A_inv = (sum gamma^t x_t x_t^T + lambda I)^-1
        # The update step:
        # A_new_inv = (1/g) * (A_old_inv - (A_old_inv x) (A_old_inv x)^T / (g + x^T A_old_inv x))
        # Wait, the standard EW-RL update for the inverse is:
        # If P_t = (lambda I + sum gamma^k x_k x_k^T)^-1
        # P_{t+1} = (1/gamma) * (P_t - (P_t x)(P_t x)^T / (gamma + x^T P_t x))
        
        x_over_g = x / g
        Ax = inv @ x_over_g
        denom = 1.0 + x_over_g @ Ax
        
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse directly if singular
            self.A[arm] = g * self.A[arm] + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.b[arm] = g * self.b[arm] + reward * x
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            return
        
        # Update inverse
        new_inv = (inv - np.outer(Ax, Ax) / denom) / g
        self.A_inv[arm] = new_inv
        
        # Update b
        self.b[arm] = g * self.b[arm] + reward * x
        
        # Update theta
        self.theta[arm] = new_inv @ self.b[arm]
        
        # Update A for fallback (optional but good for debugging/stability checks)
        self.A[arm] = g * self.A[arm] + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return the arm with highest predicted reward (greedy fallback)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                s = self.theta[a] @ context
                if s > best_score:
                    best_score = s
                    best_arm = a
            return best_arm
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            theta = self.theta[a]
            inv = self.A_inv[a]
            
            # Mean prediction
            mean = theta @ context
            
            # Variance estimate for UCB
            var = context @ inv @ context
            
            # Visit-based scaling for exploration bonus
            # Reduces bonus for well-visited arms to encourage exploitation
            # Use sqrt scaling for smoother transition
            visit_bonus_scale = 1.0 / np.sqrt(1.0 + 0.1 * self.Visits[a])
            
            bonus = self.alpha * visit_bonus_scale * np.sqrt(max(var, 0.0))
            scores.append(mean + bonus)
        
        scores = np.array(scores)
        
        # Epsilon-greedy: initial exploration (decaying)
        if self.round_count < 200:
            eps = 0.15 * (1.0 - self.round_count / 200.0)
            if np.random.random() < eps:
                return int(np.random.randint(self.n))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Calculate innovation (prediction error) before updating the model
        pred = self.theta[arm] @ context
        innovation = reward - pred
        innov_sq = innovation * innovation
        
        # Update EMA of squared innovation
        # This acts as a proxy for the process noise / non-stationarity level
        self.innovation_sq[arm] = 0.8 * self.innovation_sq[arm] + 0.2 * innov_sq
        
        # Adaptive forgetting (per-arm):
        # High innovation -> High uncertainty about current weights -> Lower gamma (faster forgetting)
        # Low innovation -> Stable environment -> Higher gamma (slower forgetting, more stability)
        
        err_scale = np.sqrt(self.innovation_sq[arm])
        
        # Map error scale to target gamma
        # Range [0.82, 0.98]
        # tanh ensures smooth saturation
        target_gamma = 0.98 - 0.16 * np.tanh(err_scale / 0.4)
        
        # Smoothly adapt gamma to avoid abrupt changes
        self.gamma[arm] = 0.80 * self.gamma[arm] + 0.20 * target_gamma
        self.gamma[arm] = np.clip(self.gamma[arm], 0.82, 0.98)
        
        # Perform the model update with the adapted gamma
        self._sherman_morrison_update(arm, context, reward, self.gamma[arm])
        
        # Track visits
        self.Visits[arm] += 1
        
        # Adjust exploration: reduce alpha over time
        # Slow decay to maintain some exploration in non-stationary settings
        self.alpha = max(0.5, self.alpha * 0.99995)
# EVOLVE-BLOCK-END
