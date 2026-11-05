import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Forgetting factor: controls how quickly old data is forgotten
        # A value of 0.95-0.99 is typical. 0.95 is responsive.
        self.gamma = 0.95
        
        # Noise variance for RLS
        self.noise_var = 1.0
        
        # Initialize per-arm parameters
        # P: Precision matrix (inverse of covariance). Initialized to identity (uniform prior)
        # b: Vector such that w = P @ b. Initialized to zeros.
        # count: Number of observations for this arm (for exploration scaling)
        
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0] * n_arms
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Exploration parameter for UCB
        self.alpha = 1.0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy using current means
            context_vec = np.array(context)
            scores = [self.b[a] @ context_vec for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        scores = []
        for a in range(self.n):
            # Mean estimate: w = P @ b
            w = self.P[a] @ self.b[a]
            
            # Predicted reward
            mean_pred = w @ context_vec
            
            # Uncertainty estimate: sqrt(x^T P x)
            # P is precision, so x^T P x is variance of the prediction if P was covariance.
            # Wait, in RLS, P is usually defined as the inverse of the covariance of the weight estimate.
            # If P = (sum gamma^t x_t x_t^T)^-1, then Cov(w) = P * noise_var.
            # So variance of prediction = x^T Cov(w) x = noise_var * x^T P x.
            # Standard deviation = sqrt(noise_var * x^T P x).
            
            # However, note that our P is initialized to I.
            # If we start with P = I, and update P = (gamma * P^-1 + x x^T / noise)^-1 ...
            # Actually, let's stick to the standard RLS formulation where we maintain P = (A)^-1.
            # Update rule for P: P_new = (1/gamma) * (P_old - P_old x (x^T P_old x + gamma/noise)^-1 x^T P_old)
            # This is the Sherman-Morrison-Woodbury update.
            
            # Let's re-verify the UCB term.
            # If P is the inverse of the covariance of the weight estimate (times noise),
            # then Cov(w) = P * noise_var.
            # Var(w^T x) = x^T Cov(w) x = noise_var * (x^T P x).
            
            var_pred = self.noise_var * (context_vec @ (self.P[a] @ context_vec))
            std_pred = np.sqrt(max(var_pred, 0.0))
            
            # UCB score
            # Scale exploration by alpha. 
            # As count increases, we might want to reduce exploration, but UCB naturally decreases
            # as P shrinks (which it does with more data).
            ucb_score = mean_pred + self.alpha * std_pred
            
            scores.append(ucb_score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context_vec = np.array(context)
        
        # Update P and b for the selected arm using RLS with forgetting factor
        # We maintain P = (sum_{i=1}^t gamma^{t-i} x_i x_i^T / noise_var)^-1
        # and b = P @ w, where w is the estimate.
        
        # Standard RLS update with forgetting factor gamma:
        # Let K be the gain vector.
        # K = P_old @ x / (gamma + x^T @ P_old @ x)
        # w_new = w_old + K @ (reward - x^T @ w_old)
        # P_new = (1/gamma) * (P_old - K @ x^T @ P_old)
        
        # Note: The factor (1/gamma) in P_new update is crucial for the forgetting mechanism.
        # Also, the denominator in K uses gamma + x^T P x.
        
        P_old = self.P[arm]
        b_old = self.b[arm]
        
        # Current weight estimate
        w_old = P_old @ b_old
        
        # Prediction
        y_pred = w_old @ context_vec
        
        # Error
        error = reward - y_pred
        
        # Gain vector K
        # K = P_old @ x / (gamma + x^T P_old x)
        Px = P_old @ context_vec
        denom = self.gamma + context_vec @ Px
        
        if denom < 1e-8:
            denom = 1e-8
            
        K = Px / denom
        
        # Update weight estimate (implicitly via b, since w = P @ b)
        # w_new = w_old + K @ error
        # But we store b such that w = P @ b.
        # So b_new = P_new^-1 @ w_new? No, that's expensive.
        # Let's derive b update directly.
        # w_new = w_old + K * error
        # We know P_new = (1/gamma) * (P_old - outer(K, Px))
        # This is getting complicated to update b directly if P changes non-trivially.
        
        # Alternative: Update w explicitly, then update P, then compute b = P^-1 w?
        # Inverting P is O(d^3). We want to avoid that.
        
        # Let's look at the b update in RLS.
        # b = P @ w.
        # w_new = w_old + K * error
        # P_new = (1/gamma) * (P_old - K Px^T)
        
        # Is there a direct update for b?
        # b_new = P_new @ w_new
        # This still requires matrix-vector multiplication P_new @ w_new, which is O(d^2).
        # But we need P_new first.
        
        # So:
        # 1. Compute K (O(d^2))
        # 2. Compute w_new = w_old + K * error (O(d))
        # 3. Compute P_new = (1/gamma) * (P_old - outer(K, Px)) (O(d^2))
        # 4. Compute b_new = P_new @ w_new (O(d^2))
        
        w_new = w_old + K * error
        
        # Update P
        # P_new = (1/gamma) * (P_old - np.outer(K, Px))
        P_new = (1.0 / self.gamma) * (P_old - np.outer(K, Px))
        
        # Ensure symmetry (numerical stability)
        P_new = (P_new + P_new.T) / 2.0
        
        # Update b
        b_new = P_new @ w_new
        
        self.P[arm] = P_new
        self.b[arm] = b_new
        self.count[arm] += 1
