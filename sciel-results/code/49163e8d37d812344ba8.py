import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # RBF kernel parameters
        # Use multiple bandwidths for multi-scale representation
        self.rbf_bandwidths = [0.5, 1.0, 2.0, 4.0, 8.0]
        self.n_rbf = len(self.rbf_bandwidths)
        self.feature_dim = dim * self.n_rbf  # Total feature dimension after RBF mapping
        
        # Model parameters for each arm in feature space
        # A: D x D matrix, b: D vector where D = feature_dim
        self.A = [np.eye(self.feature_dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(self.feature_dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting factors
        self.lambda_min = 0.95
        self.lambda_max = 0.9999
        self.lambdas = [0.9999 for _ in range(n_arms)]
        
        # Error tracking for adaptive lambda
        self.err_ema = [0.0 for _ in range(n_arms)]
        self.err_alpha = 0.05
        
        # Exploration parameter
        self.alpha_start = 1.5
        self.alpha_min = 0.3
        self.decay_rate = 0.0005
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0
        
        # Precomputed RBF centers (adaptive)
        # We'll use the contexts seen so far as centers, but limit to recent ones
        self.rbf_centers = [None for _ in range(n_arms)]
        self.max_centers = 50  # Limit number of RBF centers per arm
        self.center_update_interval = 10  # Update centers every N steps

    def _rbf_map(self, context):
        """Map context to RBF feature space using fixed bandwidths."""
        context = np.asarray(context, dtype=np.float64)
        features = []
        for bw in self.rbf_bandwidths:
            # For each dimension, apply RBF kernel with bandwidth bw
            # This creates a local basis around each context value
            rbf_vals = np.exp(-0.5 * (context / bw) ** 2)
            features.append(rbf_vals)
        return np.concatenate(features)
    
    def _solve(self, A, b):
        """Solve linear system with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_inverse_solve(self, A, vec):
        """Solve A x = vec."""
        try:
            return np.linalg.solve(A, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, vec, rcond=None)[0]

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n_arms):
                theta = self._solve(self.A[a], self.b[a])
                phi = self._rbf_map(context)
                mu = theta @ phi
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha
        alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        
        best_arm = 0
        best_score = -np.inf
        phi = self._rbf_map(context)
        
        for a in range(self.n_arms):
            A = self.A[a]
            b = self.b[a]
            
            theta = self._solve(A, b)
            mu = theta @ phi
            
            # Uncertainty term: phi^T A^{-1} phi
            A_inv_phi = self._get_inverse_solve(A, phi)
            v = phi @ A_inv_phi
            
            # UCB Score
            score = mu + alpha * np.sqrt(max(v, 0.0))
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        phi = self._rbf_map(context)
        
        # 1. Compute prediction and residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ phi
        
        # Uncertainty for reset check
        A_inv_phi = self._get_inverse_solve(A, phi)
        v = phi @ A_inv_phi
        std_pred = np.sqrt(max(v, 0.0)) + 0.5 # Add baseline noise
        
        residual = reward - mu_pred
        sq_error = residual ** 2
        
        # 2. Update Per-Arm Lambda
        # Update Error EMA
        self.err_ema[arm] = (1 - self.err_alpha) * self.err_ema[arm] + self.err_alpha * sq_error
        
        # Map Error EMA to Lambda
        # High error -> Low lambda (fast forgetting)
        # Low error -> High lambda (slow forgetting)
        
        # Normalize error with soft scaling
        err_norm = self.err_ema[arm] / (self.err_ema[arm] + 0.5)
        
        # Map [0, 1] to [lambda_max, lambda_min]
        new_lambda = self.lambda_max - (self.lambda_max - self.lambda_min) * err_norm
        
        # Smooth the lambda update to avoid oscillation
        self.lambdas[arm] = 0.9 * self.lambdas[arm] + 0.1 * new_lambda
        
        lam = self.lambdas[arm]
        
        # 3. Check for Shock Reset
        # If residual is significantly larger than expected standard deviation
        if abs(residual) > 4.0 * std_pred and self.t > 10:
            # Reset model for this arm to prioritize the new data point
            self.A[arm] = np.eye(self.feature_dim) * 1.0
            self.b[arm] = reward * phi
            self.err_ema[arm] = 0.0
            self.lambdas[arm] = self.lambda_min
        else:
            # Standard RLS update with per-arm forgetting in feature space
            self.A[arm] = lam * self.A[arm] + np.outer(phi, phi)
            self.b[arm] = lam * self.b[arm] + reward * phi
