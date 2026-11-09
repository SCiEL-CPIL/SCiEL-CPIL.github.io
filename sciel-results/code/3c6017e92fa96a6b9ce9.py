"""
Refined Non-stationary Linear Contextual Bandit Policy.
Uses Per-Arm Drift Detection with Exponentially Weighted Moving Averages (EWMA)
for adaptive forgetting factors in Recursive Least Squares (RLS).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Non-stationary LinUCB with Per-Arm Drift Detection.
    
    Features:
    1. Tracks residual EWMA per arm to detect local weight changes.
    2. Adapts forgetting factor per arm based on local drift signal.
    3. Handles 'stale' arms by applying a minimum forgetting boost if idle.
    4. Uses adaptive noise baseline for more robust drift detection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0 

        # Drift Detection State (Per-Arm)
        self.residual_ewma = [1.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.15  # Balanced responsiveness
        self._current_drift_levels = [0.0] * n_arms
        
        # Adaptive noise baseline (global EWMA of residuals when drift is low)
        self.noise_baseline = 1.0
        self.noise_baseline_alpha = 0.05
        
        # Last time each arm was played
        self.last_played = [-1000] * n_arms
        
        # Forgetting Factor Parameters
        self.min_lambda = 0.88
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.0
        self.alpha_drift_boost = 1.2
        
        # Counters
        self.counts = [0] * n_arms
        
        # Staleness threshold
        self.stale_threshold = 8

    def select(self, context):
        # Deadline Guard
        if time.time() > self.deadline:
            # Fallback: Greedy based on current theta
            context = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Compute global drift metric for alpha adaptation
        # Use soft-max (log-sum-exp) for smoother aggregation
        drift_levels = np.array(self._current_drift_levels)
        # Soft max with temperature 0.3 for smoothness
        max_drift = np.max(drift_levels)
        log_sum_exp = max_drift + np.log(np.sum(np.exp(drift_levels - max_drift)))
        global_drift = np.clip(log_sum_exp / self.n, 0.0, 1.0)
        
        # Adapt Alpha (Exploration Bonus)
        # Slower decay to maintain exploration longer
        decay_factor = 1.0 / (1.0 + 0.005 * self.t)
        base_alpha_decayed = self.base_alpha * decay_factor
        drift_boost_decayed = self.alpha_drift_boost * (1.0 / (1.0 + 0.003 * self.t))
        
        alpha = base_alpha_decayed + drift_boost_decayed * global_drift
        
        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            try:
                # Solve for theta
                theta = np.linalg.solve(A, b)
                mu = theta @ context
                
                # Compute Variance: x^T A^{-1} x
                v_vec = np.linalg.solve(A, context)
                variance = context @ v_vec
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            # Add a small boost to alpha if the arm is stale
            staleness = self.t - self.last_played[a]
            if staleness > self.stale_threshold:
                # Gradual increase in exploration for stale arms
                staleness_boost = 1.0 + 0.05 * min(staleness - self.stale_threshold, 20)
                alpha_arm = alpha * staleness_boost
            else:
                alpha_arm = alpha

            ucb_score = mu + alpha_arm * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update
        if time.time() > self.deadline:
            return

        # 1. Compute Prediction Error
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
            mu_pred = theta @ context
        except np.linalg.LinAlgError:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        sq_residual = residual ** 2
        
        # 2. Update EWMA of squared residuals
        self.residual_ewma[arm] = (1 - self.ewma_alpha) * self.residual_ewma[arm] + self.ewma_alpha * sq_residual
        
        # Update global noise baseline (only when drift is low for this arm)
        if self._current_drift_levels[arm] < 0.3:
            self.noise_baseline = (1 - self.noise_baseline_alpha) * self.noise_baseline + \
                                   self.noise_baseline_alpha * sq_residual
        
        # 3. Compute Drift Signal for this arm
        # Use adaptive noise baseline for normalization
        # If EWMA is close to noise_baseline, drift is low.
        # If EWMA is much higher, drift is high.
        # Use ratio-based detection with smooth mapping
        ratio = self.residual_ewma[arm] / max(self.noise_baseline, 0.1)
        # Map ratio to drift level: ratio=1 -> drift=0, ratio=3 -> drift=1
        drift_level = (ratio - 1.0) / 2.0
        drift_level = float(np.clip(drift_level, 0.0, 1.0))
        
        # Smooth the drift level to avoid oscillation (slower smoothing for stability)
        self._current_drift_levels[arm] = 0.85 * self._current_drift_levels[arm] + 0.15 * drift_level
        
        # 4. Compute Forgetting Factor
        # High drift -> Lower Lambda
        # Also apply staleness boost: if arm was not played recently, force some forgetting
        staleness = self.t - self.last_played[arm]
        if staleness > self.stale_threshold:
            # Apply a minimum drift level for stale arms
            stale_drift_boost = min(0.4, (staleness - self.stale_threshold) * 0.03)
            effective_drift = max(self._current_drift_levels[arm], stale_drift_boost)
        else:
            effective_drift = self._current_drift_levels[arm]
            
        # Smooth transition for lambda
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * effective_drift
        
        # 5. Update RLS State
        # A <- lam * A + x x^T
        # b <- lam * b + r x
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        self.counts[arm] += 1
        self.last_played[arm] = self.t

# EVOLVE-BLOCK-END
