"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Global Linear Bandit with Context-Augmented Features and RLS.
    
    Strategy:
    1. Instead of maintaining separate models for each arm, we use a single global model.
    2. The feature vector for arm 'a' and context 'x' is [one_hot(a), x].
    3. This allows the model to share statistical strength across arms (via the context part) 
       while still allowing arm-specific biases (via the one-hot part).
    4. We use Recursive Least Squares (RLS) with exponential forgetting for fast, stable updates.
    5. We use UCB (Upper Confidence Bound) for exploration, which is computationally cheaper 
       than Thompson Sampling (no sampling from multivariate normal) and often more robust 
       in non-stationary settings when tuned correctly.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.total_dim = n_arms + dim
        
        # Initialize RLS matrices
        # P is the inverse of the Hessian (covariance of weights)
        # lambda is the forgetting factor
        self.forgetting = 0.99
        self.regularization = 1.0  # Initial regularization to avoid singularity
        
        self.P = np.eye(self.total_dim) * self.regularization
        self.theta = np.zeros(self.total_dim)
        
        # Precompute one-hot vectors for speed
        self.one_hots = [np.zeros(n_arms) for _ in range(n_arms)]
        for i in range(n_arms):
            self.one_hots[i][i] = 1.0
            
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0
        
        # UCB exploration parameter
        # We will adapt this slightly or keep it fixed. 
        # A standard UCB is sqrt(log(t) * P). But in non-stationary, log(t) grows too fast.
        # We use a constant exploration bonus scaled by the uncertainty, or a decaying one.
        # Let's use a fixed alpha for the UCB term to keep it simple and responsive.
        self.alpha = 1.0

    def _make_feature(self, arm, context):
        """
        Create the augmented feature vector: [one_hot(arm), context]
        """
        # context is expected to be a 1D array of size self.d
        # one_hot is size self.n
        # Result is size self.n + self.d
        return np.concatenate((self.one_hots[arm], context))

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: return the arm with highest mean prediction (greedy)
            # Or just return 0 to be safe if computation is too slow
            # Given the small dimensions, this should rarely trigger.
            return 0

        context = np.asarray(context, dtype=float)
        
        scores = np.zeros(self.n)
        
        # Compute predictions and uncertainties for all arms
        # We can vectorize this somewhat, but n is small (5), so loop is fine.
        for a in range(self.n):
            x_aug = self._make_feature(a, context)
            
            # Mean prediction
            mu = self.theta @ x_aug
            
            # Variance of prediction: x^T P x
            # P is symmetric positive definite (ideally)
            var = x_aug @ (self.P @ x_aug)
            
            # UCB score: mu + alpha * sqrt(var)
            # In non-stationary settings, we might want to dampen the exploration 
            # as we get more data, but with forgetting, the effective sample size is bounded.
            # So a constant alpha is often appropriate.
            scores[a] = mu + self.alpha * np.sqrt(max(var, 0.0))
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        x_aug = self._make_feature(arm, context)
        
        # RLS Update Step
        # 1. Compute gain
        # P_new = (1/lambda) * P - (1/lambda^2) * (P x) (x^T P) / (1/lambda + x^T P x)
        # This is the standard RLS with forgetting.
        
        # For numerical stability and efficiency, we use the matrix inversion lemma form.
        # Let P_old be self.P
        # x is x_aug
        # y is reward
        
        # Step 1: Compute Px
        Px = self.P @ x_aug
        
        # Step 2: Compute denominator
        # denom = 1/lambda + x^T P x
        # Note: x^T P x = x^T Px
        denom = (1.0 / self.forgetting) + (x_aug @ Px)
        
        # Step 3: Compute gain vector K
        # K = Px / denom
        # Actually, the standard RLS gain is K = P x / (lambda + x^T P x)
        # Wait, let's derive carefully.
        # Minimize J = sum lambda^k (y_k - x_k^T theta)^2
        # Gradient: -2 sum lambda^k x_k (y_k - x_k^T theta) = 0
        # (sum lambda^k x_k x_k^T) theta = sum lambda^k x_k y_k
        # Let A = sum lambda^k x_k x_k^T, b = sum lambda^k x_k y_k
        # theta = A^{-1} b = P b
        # Recurrence:
        # A_t = (1/lambda) A_{t-1} + x_t x_t^T
        # b_t = (1/lambda) b_{t-1} + x_t y_t
        # P_t = A_t^{-1}
        # Using matrix inversion lemma:
        # P_t = ( (1/lambda) A_{t-1} + x x^T )^{-1}
        # = ( (1/lambda) (A_{t-1} + lambda x x^T) )^{-1}
        # = lambda (A_{t-1} + lambda x x^T)^{-1}
        # Let M = A_{t-1} + lambda x x^T
        # M^{-1} = A_{t-1}^{-1} - A_{t-1}^{-1} x x^T A_{t-1}^{-1} / (1 + x^T A_{t-1}^{-1} x)
        # = P_{t-1} - P_{t-1} x x^T P_{t-1} / (1 + x^T P_{t-1} x)
        # So P_t = lambda * [ P_{t-1} - P_{t-1} x x^T P_{t-1} / (1 + x^T P_{t-1} x) ]
        # P_t = lambda P_{t-1} - lambda^2 P_{t-1} x x^T P_{t-1} / (1 + x^T P_{t-1} x)
        
        # Let's check the denominator in the formula above.
        # denom = 1 + x^T P_{t-1} x
        # In my previous thought, I had (1/lambda + x^T P x). Let's re-verify.
        # The standard RLS update for P is:
        # P_t = (1/lambda) P_{t-1} - (1/lambda) * (P_{t-1} x) (x^T P_{t-1}) / (lambda + x^T P_{t-1} x)
        # Let's check dimensions and limits.
        # If lambda=1, P_t = P_{t-1} - P_{t-1} x x^T P_{t-1} / (1 + x^T P_{t-1} x). This matches standard RLS.
        # So the formula is:
        # P_t = (1/lambda) P_{t-1} - (1/lambda) * (P_{t-1} x) (x^T P_{t-1}) / (lambda + x^T P_{t-1} x)
        
        # Let's implement this.
        # Px = P_{t-1} x
        # x^T Px = x^T P_{t-1} x
        # denom = lambda + x^T Px
        
        lambda_val = self.forgetting
        denom = lambda_val + (x_aug @ Px)
        
        # Gain vector K = (1/lambda) * Px / denom
        K = (1.0 / lambda_val) * (Px / denom)
        
        # Update P
        # P_t = (1/lambda) P_{t-1} - K x^T P_{t-1}  (Note: K x^T P is rank-1 update)
        # Actually, P_t = (1/lambda) P_{t-1} - K (x^T P_{t-1})
        # Since K is a vector, K (x^T P) is outer product of K and (P x)^T?
        # Let's look at the term: (P x) (x^T P) is a matrix.
        # The term is (1/lambda) * (P x) (x^T P) / denom
        # = K (x^T P)
        # Since x^T P = (P x)^T = Px^T
        # So the term is K Px^T
        
        self.P = (1.0 / lambda_val) * self.P - np.outer(K, Px)
        
        # Update theta
        # theta_t = theta_{t-1} + K (y - x^T theta_{t-1})
        # Prediction error
        pred = self.theta @ x_aug
        error = reward - pred
        
        self.theta += K * error
        
        self.t += 1
# EVOLVE-BLOCK-END
