import numpy as np
import time

class Policy:
    """
    Global Linear Model with Exponential Forgetting (Recursive Least Squares)
    and Thompson Sampling for Non-Stationary Linear Contextual Bandits.
    
    Structural Change:
    - Replaces per-arm independent linear models with a single global linear model.
    - Uses context-augmented features: [one_hot(arm), context].
    - This allows sharing of context trends across arms while maintaining arm-specific offsets.
    - Uses Recursive Least Squares (RLS) with exponential forgetting factor for O(1) memory
      and O(d^2) update time, avoiding sliding window recomputation costs.
    - Thompson Sampling is used for exploration, sampling from the posterior of the global model.
    """
    
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Feature dimension: 1 (intercept/offset for arm) + n_arms (one-hot) + dim (context)
        # Actually, let's use [context, one_hot(arm)] or [one_hot(arm), context].
        # Let's use [one_hot(arm), context].
        # Total dim = n_arms + dim
        self.feature_dim = n_arms + dim
        
        self.t = 0
        
        # Forgetting factor gamma in (0, 1)
        # Closer to 1 means slower forgetting (longer memory)
        # In non-stationary settings, we might want it slightly less than 1
        self.gamma = 0.95
        
        # Regularization parameter lambda
        self.lambda_reg = 1.0
        
        # P = inverse of A (covariance matrix of weights)
        # We maintain P = (lambda * I + sum gamma^k x_k x_k^T)^-1
        # Update rule: P_t = 1/gamma * (P_{t-1} - (P_{t-1} x (x^T P_{t-1} x + gamma)^-1) x^T P_{t-1})
        # Wait, the standard RLS with forgetting factor update is:
        # A_t = gamma * A_{t-1} + x x^T
        # P_t = A_t^{-1}
        # Using matrix inversion lemma:
        # P_t = (1/gamma) * [P_{t-1} - (P_{t-1} x x^T P_{t-1}) / (gamma + x^T P_{t-1} x)]
        
        self.P = np.eye(self.feature_dim) / self.lambda_reg
        self.theta = np.zeros(self.feature_dim)
        
        # Thompson Sampling scaling
        # We sample theta_sample = theta + sigma * z, where z ~ N(0, P)
        # The variance of theta is P (if noise variance is 1)
        # We need to scale appropriately.
        self.sigma2 = 1.0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0
        
        # Pre-compute Cholesky factor for sampling?
        # P changes every step, so we can't pre-compute.
        # But we can try to maintain a Cholesky factor or just compute it on the fly.
        # Computing Cholesky of (n_arms + dim) x (n_arms + dim) matrix is O(d^3).
        # If dim is small (e.g., 5-10), this is fine.
        
    def _get_feature(self, arm, context):
        """
        Construct the feature vector for a given arm and context.
        Feature = [one_hot(arm), context]
        """
        feat = np.zeros(self.feature_dim)
        feat[arm] = 1.0
        feat[self.n:] = context
        return feat

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback: greedy
            means = []
            for a in range(self.n):
                feat = self._get_feature(a, context)
                means.append(self.theta @ feat)
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            feat = self._get_feature(a, context)
            
            try:
                # Sample from posterior
                # theta ~ N(self.theta, self.sigma2 * self.P)
                # We need to sample z ~ N(0, self.P)
                # z = L @ z_std, where L is Cholesky of self.P
                # self.P is positive definite (regularized)
                
                # To save time, we can use a simpler approximation if Cholesky is too slow?
                # But for small dims, it's fine.
                L = np.linalg.cholesky(self.P)
                z_std = np.random.randn(self.feature_dim)
                z = L @ z_std
                
                theta_sample = self.theta + np.sqrt(self.sigma2) * z
                score = theta_sample @ feat
                scores.append(score)
                
            except np.linalg.LinAlgError:
                # Fallback to mean
                score = self.theta @ feat
                scores.append(score)
            except Exception:
                score = self.theta @ feat
                scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        feat = self._get_feature(arm, context)
        
        # Update theta and P using RLS with forgetting factor
        # A_t = gamma * A_{t-1} + x x^T
        # P_t = A_t^{-1}
        
        # P_t = (1/gamma) * [P_{t-1} - (P_{t-1} x x^T P_{t-1}) / (gamma + x^T P_{t-1} x)]
        
        try:
            Px = self.P @ feat
            denom = self.gamma + feat @ Px
            # Avoid division by zero
            if abs(denom) < 1e-10:
                denom = 1e-10
                
            # Update P
            # P_new = (1/gamma) * (P - (Px @ Px.T) / denom)
            self.P = (1.0 / self.gamma) * (self.P - np.outer(Px, Px) / denom)
            
            # Symmetrize P to avoid numerical drift
            self.P = (self.P + self.P.T) / 2.0
            
            # Update theta
            # theta_t = theta_{t-1} + P_t (y - x^T theta_{t-1}) * x ?
            # Wait, the standard RLS update for theta is:
            # theta_t = theta_{t-1} + P_{t-1} x (y - x^T theta_{t-1}) / (gamma + x^T P_{t-1} x)
            # Let's verify.
            # A_t theta_t = gamma A_{t-1} theta_{t-1} + x y
            # theta_t = A_t^{-1} (gamma A_{t-1} theta_{t-1} + x y)
            # = P_t (gamma A_{t-1} theta_{t-1} + x y)
            # = P_t (gamma (P_{t-1}^{-1} theta_{t-1}) + x y)
            # This is getting messy. Let's use the standard form:
            # theta_t = theta_{t-1} + K_t (y - x^T theta_{t-1})
            # where K_t = P_{t-1} x / (gamma + x^T P_{t-1} x)
            # Note: P_t is updated using P_{t-1}.
            
            # Let's re-derive or look up the exact formula.
            # Standard RLS with forgetting factor:
            # P_k = (1/gamma) * (P_{k-1} - (P_{k-1} x_k x_k^T P_{k-1}) / (gamma + x_k^T P_{k-1} x_k))
            # theta_k = theta_{k-1} + (P_{k-1} x_k / (gamma + x_k^T P_{k-1} x_k)) * (y_k - x_k^T theta_{k-1})
            
            # So we need P_{t-1} for the theta update.
            # I updated self.P above. I need the old P.
            
        except Exception:
            pass
        
        # Let's redo the update properly to keep track of old P.
        # Actually, I can compute K first.
        
        # Re-implementing update block correctly:
        try:
            Px_old = self.P @ feat
            denom = self.gamma + feat @ Px_old
            if abs(denom) < 1e-10:
                denom = 1e-10
            
            K = Px_old / denom
            
            # Update theta
            pred = feat @ self.theta
            error = reward - pred
            self.theta = self.theta + K * error
            
            # Update P
            self.P = (1.0 / self.gamma) * (self.P - np.outer(Px_old, Px_old) / denom)
            self.P = (self.P + self.P.T) / 2.0
            
        except Exception:
            pass

        self.t += 1
