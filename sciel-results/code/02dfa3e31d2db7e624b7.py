import numpy as np
import time

class Policy:
    """
    Global Linear Model with Arm Biases for Non-Stationary Contextual Bandits.
    
    Strategy:
    Instead of maintaining n_arms separate linear models (each with dim parameters),
    we maintain a single global linear model (dim parameters) and n_arms scalar biases.
    This reduces the total parameter count from n_arms * dim to dim + n_arms.
    
    Hypothesis:
    In many contextual bandit settings, the context features influence the baseline
    reward similarly across arms, while the arm choice adds a specific offset.
    Even if this is not strictly true, sharing the weight vector allows the model
    to learn the context-reward relationship faster (using data from all arms),
    which is crucial in non-stationary environments where data is scarce per arm.
    
    The prediction for arm a given context x is:
    r_hat(a, x) = x^T w + b_a
    
    We use a forgetting factor lambda to handle non-stationarity.
    We use UCB with a global uncertainty term derived from the global covariance.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global Ridge Regression State
        # A: dim x dim matrix
        # b: dim vector
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        self.inv_A = np.eye(dim)
        
        # Arm-specific biases
        # We track the mean bias and the count for each arm to handle forgetting
        # For simplicity and robustness, we use a decaying sum approach for biases too
        # bias_sum[a] = sum(lambda^t * reward_t * indicator(arm_t=a))
        # bias_count[a] = sum(lambda^t * indicator(arm_t=a))
        # b_a = bias_sum[a] / bias_count[a]
        
        # However, to integrate with the global model update which subtracts the bias,
        # we need to be careful.
        
        # Let's use a simpler formulation for the bias that also forgets.
        # We will update the global model using "corrected" rewards: r' = r - b_a
        # But b_a itself needs to be estimated.
        
        # Alternative: Joint optimization.
        # Minimize sum (r_t - x_t^T w - b_{a_t})^2 + lambda_regularization
        # This is a linear regression problem with features [x_t, 1_{a_t}]
        # But the dimension of the feature vector would be dim + n_arms.
        # Let's implement this directly as a single linear model with augmented features.
        
        # Feature vector z_t = [x_t, e_{a_t}] where e_{a_t} is one-hot for arm.
        # Dimension D = dim + n_arms
        # We maintain A_global (D x D) and b_global (D)
        
        self.D = dim + n_arms
        self.A_global = np.eye(self.D)
        self.b_global = np.zeros(self.D)
        self.inv_A_global = np.eye(self.D)
        
        # Forgetting factor
        # We start with a moderate forgetting factor.
        # Adaptive forgetting: if prediction error is high, forget faster.
        self.lambda_val = 0.95
        self.lambda_min = 0.85
        self.lambda_max = 0.99
        
        # Exploration parameter
        self.alpha = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.5
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0
        self.t = 0
        
        # Error tracking for adaptive lambda
        self.recent_errors = []
        self.error_window_size = 20

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_augmented_context(self, context, arm):
        """
        Construct the augmented feature vector [x, one_hot_arm].
        """
        z = np.zeros(self.D)
        z[:self.d] = context
        z[self.d + arm] = 1.0
        return z

    def _update_lambda(self, error):
        """
        Adapt forgetting factor based on recent prediction errors.
        High error -> lower lambda (forget faster).
        Low error -> higher lambda (forget slower).
        """
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window_size:
            self.recent_errors.pop(0)
        
        if len(self.recent_errors) >= 5:
            mse = np.mean(np.array(self.recent_errors)**2)
            # Map MSE to lambda
            # If MSE is small, lambda close to max. If MSE large, lambda close to min.
            # Using a sigmoid-like scaling
            scale = 0.1
            norm_mse = mse / (mse + scale)
            self.lambda_val = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
            self.lambda_val = max(self.lambda_min, min(self.lambda_max, self.lambda_val))

    def _update_alpha(self):
        """
        Adapt exploration parameter.
        """
        if len(self.recent_errors) >= 10:
            errors = np.array(self.recent_errors)
            mse = np.mean(errors**2)
            # Target alpha based on MSE
            target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.2))
            # Smooth update
            self.alpha = 0.8 * self.alpha + 0.2 * target_alpha
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick best mean
            scores = []
            for a in range(self.n):
                z = self._get_augmented_context(context, a)
                mu = z @ self.b_global # Wait, theta = inv_A @ b. We need theta.
                # We haven't computed theta in select. We should cache it or compute on fly.
                # Computing theta on fly is O(D^2). With D ~ dim + 5, it's small.
                # But let's cache theta.
                pass
            
            # Recompute theta for fallback? Or just return 0?
            # Better to maintain a cached theta in update.
            if hasattr(self, 'theta_global'):
                theta = self.theta_global
            else:
                theta = self.inv_A_global @ self.b_global
            
            scores = []
            for a in range(self.n):
                z = self._get_augmented_context(context, a)
                mu = z @ theta
                scores.append(mu)
            return int(np.argmax(scores))

        # Compute scores for all arms
        # theta = inv_A_global @ b_global
        # We cache this in update() as self.theta_global
        if not hasattr(self, 'theta_global'):
            self.theta_global = self.inv_A_global @ self.b_global
        
        theta = self.theta_global
        
        scores = []
        for a in range(self.n):
            z = self._get_augmented_context(context, a)
            mu = z @ theta
            # Uncertainty: z^T inv_A_global z
            # Note: The one-hot part of z is sparse, so this can be computed efficiently,
            # but for small D, full matrix-vector is fine.
            var = z @ self.inv_A_global @ z
            var = max(var, 0.0)
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Prediction error
        if not hasattr(self, 'theta_global'):
            self.theta_global = self.inv_A_global @ self.b_global
        
        z = self._get_augmented_context(context, arm)
        pred = z @ self.theta_global
        error = reward - pred
        
        # 2. Update adaptive parameters
        self._update_lambda(error)
        self._update_alpha()
        
        lam = self.lambda_val
        
        # 3. Update Global Linear Model
        # A_new = lam * A_old + z z^T
        # b_new = lam * b_old + reward * z
        
        # SMW Update for inv_A
        # inv(lam * A + z z^T) = (1/lam) * inv(A + (1/lam) z z^T)
        # = (1/lam) * [A^{-1} - A^{-1} z z^T A^{-1} / (lam + z^T A^{-1} z)]
        
        inv_old = self.inv_A_global
        tmp = inv_old @ z
        denom = lam + z @ tmp
        
        if abs(denom) < 1e-10:
            # Fallback
            self.A_global = lam * self.A_global + np.outer(z, z)
            self.b_global = lam * self.b_global + reward * z
            try:
                self.inv_A_global = np.linalg.inv(self.A_global)
            except np.linalg.LinAlgError:
                self.inv_A_global = np.eye(self.D)
        else:
            # SMW
            self.inv_A_global = (1.0 / lam) * (inv_old - np.outer(tmp, tmp) / denom)
            
            # Update A and b for potential debugging or if SMW fails later
            # We don't strictly need to update A if we always use inv_A, 
            # but we do need b.
            # Actually, we don't need A if we use SMW for inv_A.
            # But we must update b.
            self.b_global = lam * self.b_global + reward * z
            
        # 4. Update cached theta
        self.theta_global = self.inv_A_global @ self.b_global

