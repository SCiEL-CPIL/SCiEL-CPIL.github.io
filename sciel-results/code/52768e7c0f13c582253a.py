"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Global Bayesian Linear Regression
    with Exponential Forgetting and Thompson Sampling.
    
    Strategy:
    1. Global Model: Instead of separate models per arm, we assume a single linear
       relationship between context and reward, parameterized by a shared weight vector 
       theta. This is valid because the task specifies "linear rewards" where the 
       context determines the reward for the selected arm. In a 5-arm linear bandit,
       typically y = x^T theta_a. However, if the arm is part of the context or if 
       we treat the arm selection as an action that reveals the reward for that 
       specific linear function, a common approach in "linear contextual bandits" 
       is to have a separate theta for each arm. 
       
       Wait, the standard "Linear Contextual Bandit" (LinUCB) assumes r = x^T theta_a.
       The current program uses separate windows per arm.
       
       Let's reconsider. If we use a single global theta, we are assuming 
       r = x^T theta for ALL arms, which is likely incorrect if arms have different 
       weights. 
       
       Alternative Structural Change: 
       The current approach uses a Sliding Window (hard cutoff). 
       A different design family is **Recursive Least Squares (RLS) with Exponential Forgetting**.
       This maintains a running inverse covariance matrix (P) and mean (theta) for 
       EACH arm, but updates them recursively with a discount factor gamma.
       
       Why is this structurally different?
       1. It does not store raw data (O(1) memory per arm vs O(W) memory).
       2. It uses a soft forgetting mechanism (exponential decay) rather than a hard 
          window drop-off.
       3. It allows for potentially faster adaptation if gamma is tuned, as it 
          effectively gives more weight to recent data smoothly.
       4. It avoids the O(W*d^2) recomputation cost in the update step, making it 
          O(d^2) per update, which is significantly faster for large W.
       
       Let's implement per-arm RLS with Exponential Forgetting and Thompson Sampling.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Exponential forgetting factor. 
        # gamma < 1 means older data is discounted.
        # 0.95 is a common choice. 0.9 is more responsive.
        # We choose 0.9 for higher responsiveness to non-stationarity.
        self.gamma = 0.9
        
        # Regularization to prevent singularity
        self.lambda_ = 1.0
        
        # Initialize per-arm parameters:
        # theta: mean estimate of weights (d,)
        # P: inverse of the effective covariance matrix (d, d)
        # Initially, P = I / lambda (prior covariance), theta = 0
        
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        self.Ps = [np.eye(dim) / self.lambda_ for _ in range(n_arms)]
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy
        if time.time() > self.deadline:
            # Simple greedy: pick arm with highest mean reward so far
            best_arm = 0
            best_val = -np.inf
            ctx = np.asarray(context, dtype=float)
            for a in range(self.n):
                val = self.thetas[a] @ ctx
                if val > best_val:
                    best_val = val
                    best_arm = a
            return best_arm

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            try:
                # Thompson Sampling:
                # Sample theta_sample ~ N(theta_a, P_a)
                # P_a is the covariance of the posterior.
                
                theta_hat = self.thetas[a]
                cov = self.Ps[a]
                
                # Ensure positive definite for Cholesky
                # Add small jitter if needed
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    cov = cov + 1e-6 * np.eye(self.d)
                    L = np.linalg.cholesky(cov)
                
                # Sample
                z = np.random.randn(self.d)
                theta_sample = theta_hat + L @ z
                
                # Compute score
                score = theta_sample @ context
                scores.append(score)
                
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                scores.append(self.thetas[a] @ context)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Recursive Least Squares Update with Exponential Forgetting
        # 
        # Standard RLS update:
        # K = P x / (lambda + x^T P x)
        # P_new = (I - K x^T) P / gamma  (Note: the division by gamma applies to the 
        #          "effective" matrix. Let's derive carefully.)
        #
        # We want to minimize:
        # sum_{t=1}^T gamma^{T-t} || y_t - x_t^T theta ||^2
        #
        # Let A_t = sum_{i=1}^t gamma^{t-i} x_i x_i^T + lambda I
        # Let b_t = sum_{i=1}^t gamma^{t-i} x_i y_i
        #
        # Recurrence:
        # A_t = gamma A_{t-1} + x_t x_t^T
        # b_t = gamma b_{t-1} + x_t y_t
        #
        # We maintain P = A^{-1}.
        # Using Matrix Inversion Lemma (Sherman-Morrison-Woodbury):
        # A_t = gamma A_{t-1} + x x^T
        # A_t^{-1} = (gamma A_{t-1} + x x^T)^{-1}
        #
        # Let M = gamma A_{t-1}. Then A_t = M + x x^T.
        # (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        #
        # Note: M^{-1} = (gamma A_{t-1})^{-1} = (1/gamma) A_{t-1}^{-1} = (1/gamma) P_{t-1}
        #
        # So:
        # P_t = (1/gamma) P_{t-1} - [ (1/gamma) P_{t-1} x x^T (1/gamma) P_{t-1} ] / (1 + x^T (1/gamma) P_{t-1} x)
        #
        # Let P_prev = P_{t-1}
        # Let K = P_prev x / (gamma + x^T P_prev x)  <-- Wait, let's simplify the denominator.
        #
        # Denominator term: 1 + (1/gamma) x^T P_prev x = (gamma + x^T P_prev x) / gamma
        #
        # Numerator term: (1/gamma^2) P_prev x x^T P_prev
        #
        # Term to subtract:
        # [ (1/gamma^2) P_prev x x^T P_prev ] / [ (gamma + x^T P_prev x) / gamma ]
        # = (1/gamma) P_prev x x^T P_prev / (gamma + x^T P_prev x)
        #
        # So:
        # P_t = (1/gamma) P_prev - (1/gamma) P_prev x x^T P_prev / (gamma + x^T P_prev x)
        # P_t = (1/gamma) [ P_prev - P_prev x x^T P_prev / (gamma + x^T P_prev x) ]
        #
        # Let K = P_prev x / (gamma + x^T P_prev x)
        # Then P_prev x x^T P_prev / (gamma + x^T P_prev x) = K x^T P_prev
        #
        # So P_t = (1/gamma) (I - K x^T) P_prev
        
        P_prev = self.Ps[arm]
        
        # Compute K
        Px = P_prev @ context
        denom = self.gamma + context @ Px
        K = Px / denom
        
        # Update P
        # P_new = (1/gamma) * (I - K x^T) * P_prev
        # To avoid allocating I, we can do:
        # P_new = (1/gamma) * P_prev - (1/gamma) * K[:, None] * (context @ P_prev)
        # context @ P_prev is a row vector (1, d)
        # K[:, None] is (d, 1)
        # Product is (d, d)
        
        P_new = (P_prev - np.outer(K, context @ P_prev)) / self.gamma
        
        # Update theta
        # theta_t = A_t^{-1} b_t
        # b_t = gamma b_{t-1} + x y
        # theta_t = P_t (gamma b_{t-1} + x y)
        # Note: P_t = A_t^{-1}.
        # We know theta_{t-1} = P_{t-1} b_{t-1} => b_{t-1} = P_{t-1}^{-1} theta_{t-1} = A_{t-1} theta_{t-1}
        # This path is messy. Let's use the standard RLS theta update formula.
        #
        # Standard RLS (without forgetting) theta_t = theta_{t-1} + K (y - x^T theta_{t-1})
        # With forgetting, the gain K is modified as derived above.
        # The update rule for theta remains:
        # theta_t = theta_{t-1} + K (y - x^T theta_{t-1})
        
        error = reward - context @ self.thetas[arm]
        self.thetas[arm] = self.thetas[arm] + K * error
        
        # Store updated P
        self.Ps[arm] = P_new
        
        self.t += 1
# EVOLVE-BLOCK-END
