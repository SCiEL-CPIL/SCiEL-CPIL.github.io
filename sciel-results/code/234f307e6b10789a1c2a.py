"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Global Linear UCB with non-stationary handling.
    
    Uses a single global linear model where the feature vector is augmented 
    with one-hot arm indicators. This allows the model to learn inter-arm 
    correlations and share context representations, improving sample efficiency 
    in non-stationary environments.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.D = dim + n_arms  # Total feature dimension
        
        # Global linear model parameters
        # A is the precision matrix (inverse covariance), b is the weighted sum of rewards * features
        self.A = np.eye(self.D)
        self.b = np.zeros(self.D)
        
        # Regularization / Forgetting factor
        # We use a dynamic lambda to handle non-stationarity
        self._base_lambda = 0.95
        self._min_lambda = 0.85
        self._max_lambda = 0.99
        self._recent_errors = []
        self._error_window = 20
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 115.0  # 5s safety margin before 120s
        
        # Exploration parameter
        self._alpha = 0.7
        
        # Cache for last prediction to compute error
        self._last_pred_mu = None
        self._last_arm = None
        self._last_context = None

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 5:
            return self._base_lambda
        
        recent = self._recent_errors[-self._error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0
        
        # High error relative to variance suggests drift -> lower lambda (forget faster)
        # Low error suggests stability -> higher lambda (retain more)
        ratio = mean_err / (std_err + 1e-8)
        
        if ratio > 2.0:
            lambda_val = self._min_lambda
        elif ratio > 1.0:
            # Linear interpolation between min and base
            lambda_val = self._min_lambda + (self._base_lambda - self._min_lambda) * (1.0 - (ratio - 1.0))
        elif ratio < 0.5:
            # Linear interpolation between base and max
            lambda_val = self._base_lambda + (self._max_lambda - self._base_lambda) * (0.5 - ratio)
        else:
            lambda_val = self._base_lambda
            
        return np.clip(lambda_val, self._min_lambda, self._max_lambda)

    def _build_feature(self, context, arm):
        """
        Construct the augmented feature vector for a given context and arm.
        Feature = [context, one_hot_arm]
        """
        # Ensure context is a 1D numpy array
        ctx = np.asarray(context, dtype=np.float64).flatten()
        # One-hot encode the arm
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        # Concatenate
        return np.concatenate([ctx, one_hot])

    def _predict_scores(self, context):
        """
        Compute UCB scores for all arms using the global model.
        Returns:
            scores: array of shape (n_arms,)
        """
        # Compute inverse of A (or use Cholesky for stability if possible)
        try:
            # Use Cholesky for better numerical stability and speed if A is SPD
            L = np.linalg.cholesky(self.A)
            # A_inv is not explicitly needed if we solve, but for UCB we need v = x^T A^{-1} x
            # We can solve A y = x for y, then v = x dot y
            # Or just invert for simplicity given small D (dim + 5)
            # Let's invert for clarity and robustness against non-SPD edge cases during evolution
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            A_inv = np.linalg.pinv(self.A)
            theta = A_inv @ self.b
            
        scores = np.zeros(self.n)
        for a in range(self.n):
            phi = self._build_feature(context, a)
            mu = theta @ phi
            # Uncertainty: phi^T A^{-1} phi
            v = phi @ A_inv @ phi
            sigma = np.sqrt(max(v, 0.0))
            scores[a] = mu + self._alpha * sigma
            
        return scores

    def select(self, context):
        self._last_context = np.asarray(context, dtype=np.float64).copy()
        
        # Deadline guard: if close to limit, use greedy (mu only)
        if time.time() > self._deadline - 2.0:
            try:
                A_inv = np.linalg.inv(self.A)
                theta = A_inv @ self.b
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A)
                theta = A_inv @ self.b
                
            scores = np.zeros(self.n)
            for a in range(self.n):
                phi = self._build_feature(self._last_context, a)
                scores[a] = theta @ phi
            return int(np.argmax(scores))
        
        scores = self._predict_scores(self._last_context)
        best_arm = int(np.argmax(scores))
        
        # Store the mean prediction of the chosen arm for error tracking
        try:
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(self.A)
            theta = A_inv @ self.b
            
        phi_best = self._build_feature(self._last_context, best_arm)
        self._last_pred_mu = theta @ phi_best
        self._last_arm = best_arm
        
        return best_arm

    def update(self, context, arm, reward):
        # Track prediction error for adaptive forgetting
        if self._last_arm == arm and self._last_pred_mu is not None and self._last_context is not None:
            # Check if context is similar enough? For simplicity, assume it is the same round's context
            err = abs(reward - self._last_pred_mu)
            self._recent_errors.append(err)
            if len(self._recent_errors) > self._error_window * 2:
                self._recent_errors = self._recent_errors[-self._error_window:]
        
        # Update global model with forgetting factor
        lam = self._get_lambda()
        
        # Construct feature vector for the observed arm
        ctx = np.asarray(context, dtype=np.float64).flatten()
        phi = self._build_feature(ctx, arm)
        
        # Update A and b
        # A_new = lam * A_old + phi phi^T
        # b_new = lam * b_old + reward * phi
        self.A = lam * self.A + np.outer(phi, phi)
        self.b = lam * self.b + reward * phi
# EVOLVE-BLOCK-END
