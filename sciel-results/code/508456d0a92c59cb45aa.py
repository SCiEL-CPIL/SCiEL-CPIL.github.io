# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy using LinUCB with
    Sherman-Morrison inverse covariance updates and adaptive forgetting.
    
    Structural improvements:
    - Maintains V = A^{-1} directly for O(d^2) updates.
    - Adaptive forgetting factor gamma based on recent reward variance.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Time management
        self._t0 = time.time()
        self._deadline = self._t0 + 145.0  # 5s safety margin before 150s kill

        # Model parameters
        # V is the inverse of the regularized covariance matrix A
        # A = sum(gamma^t * x x^T) + lambda * I
        # We initialize V = I / lambda. Let lambda = 1.0 for simplicity, so V = I.
        # Note: In standard LinUCB, A = I + sum(x x^T). Here we use a discounting factor.
        # To keep V well-conditioned, we don't add a large identity matrix initially, 
        # but the Sherman-Morrison update handles the rank-1 update.
        # However, for numerical stability, it's often good to start with a regularized inverse.
        # Let's start with V = I (equivalent to A = I).
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.999
        
        # Exploration parameter
        self.base_C = 2.0
        self.C_min = 0.5
        self.C_max = 4.0

        # Tracking for adaptation
        self.reward_history = []
        self.max_history = 100
        self.total_rounds = 0

    def _check_deadline(self):
        """Return True if we should stop exploring and just pick best known arm."""
        return time.time() > self._deadline

    def _adaptive_gamma(self):
        """
        Adapt forgetting factor based on recent reward variance.
        High variance suggests non-stationarity -> lower gamma (faster forgetting).
        """
        if len(self.reward_history) < 20:
            return self.base_gamma

        recent = self.reward_history[-min(50, len(self.reward_history)):]
        var = np.var(recent)
        mean_abs = np.mean(np.abs(recent))

        # Normalize variance relative to mean absolute reward to be scale-invariant
        if mean_abs > 1e-9:
            normalized_var = var / (mean_abs ** 2 + 1e-9)
        else:
            normalized_var = 0.0

        # Map normalized variance to gamma range
        # Low variance -> high gamma (slow forgetting), High variance -> low gamma (fast forgetting)
        # normalized_var is typically small. We scale it to map [0, 1] to [gamma_max, gamma_min]
        # If normalized_var is 0, gamma = gamma_max. If normalized_var >= 0.5, gamma = gamma_min.
        scale = min(1.0, normalized_var * 2.0)
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * scale
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """
        Adapt exploration parameter C.
        Higher C for arms with fewer samples to encourage exploration.
        Lower C overall as total rounds increase to exploit.
        """
        total_count = sum(self.counts)
        if total_count < 10:
            return self.C_max

        # Decrease exploration as we learn more globally
        progress = min(1.0, total_count / 500.0)
        C = self.C_max - (self.C_max - self.C_min) * progress

        # Increase exploration for under-explored arms
        avg_count = total_count / self.n
        if self.counts[arm] < avg_count:
            C *= 1.5
        elif self.counts[arm] > 2 * avg_count:
            C *= 0.8

        return max(self.C_min, min(self.C_max, C))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        if self._check_deadline():
            # Deadline approaching: pick the arm with highest estimated reward (greedy)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # theta = V @ b
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            
            # Uncertainty bonus: sqrt(x^T V x)
            # v = V @ x
            v = self.V[a] @ context
            uncertainty = np.sqrt(np.dot(context, v))
            
            C = self._adaptive_C(a)
            scores[a] = mu + C * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.total_rounds += 1
        
        if self._check_deadline():
            # If we are past the deadline, we should minimize computation.
            # However, the evaluator kills us at 150s. 
            # If we are close, we can skip the heavy update to save time for the next select?
            # Actually, if we are past the deadline, the next select will be greedy.
            # We can skip the update to save time, but it might hurt if we have a few more rounds.
            # Given the 5s margin, we should still update if possible, but fast.
            pass

        gamma = self._adaptive_gamma()
        
        # Sherman-Morrison update with forgetting factor
        # A_new = gamma * A_old + x x^T
        # V_new = A_new^{-1}
        #
        # Derivation:
        # A_new = gamma * (A_old + (1/gamma) x x^T)
        # V_new = (1/gamma) * (A_old + (1/gamma) x x^T)^{-1}
        # Using SM formula on (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = A_old, u = (1/gamma) x, v = x
        # (A_old + (1/gamma) x x^T)^{-1} = V_old - V_old ((1/gamma) x) x^T V_old / (1 + x^T V_old ((1/gamma) x))
        # Let Vx = V_old @ x
        # Numerator term: (1/gamma) * Vx @ x^T
        # Denominator term: 1 + (1/gamma) * x^T Vx
        #
        # V_temp = V_old - (1/gamma) * (Vx @ x^T) / (1 + (1/gamma) * x^T Vx)
        # V_new = (1/gamma) * V_temp
        # V_new = (1/gamma) * V_old - (1/gamma)^2 * (Vx @ x^T) / (1 + (1/gamma) * x^T Vx)
        
        V = self.V[arm]
        x = context
        gamma_inv = 1.0 / gamma
        
        # Compute Vx
        Vx = V @ x
        # Compute scalar x^T Vx
        xVx = np.dot(x, Vx)
        
        # Denominator
        denom = 1.0 + gamma_inv * xVx
        
        # Update V
        # V_new = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom
        # To avoid creating a new outer product array if possible, but d is small (likely 5-10), so it's fine.
        self.V[arm] = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom
        
        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * x
        
        self.counts[arm] += 1
        
        # Track reward history for adaptive gamma
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
# EVOLVE-BLOCK-END
