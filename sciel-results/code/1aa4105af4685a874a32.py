"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Global Thompson Sampling for non-stationary linear contextual bandit.
    
    Structural changes from previous approaches:
    1. Uses a single global model instead of per-arm models, assuming the 
       reward depends on the context and the chosen arm in a shared linear 
       parameter space (e.g., via one-hot encoding of the arm in the context).
    2. Replaces UCB exploration with Thompson Sampling (stochastic sampling 
       from the posterior), which is often superior in non-stationary settings 
       as it naturally balances exploration and exploitation without needing 
       to explicitly tune the optimism parameter alpha.
    3. Uses the Sherman-Morrison-Woodbury formula for O(d^2) rank-1 updates 
       of the inverse covariance matrix, which is faster and more stable 
       than repeatedly solving linear systems or inverting matrices.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.beta_forget = 0.95  # Fixed aggressive forgetting factor
        
        # Global model: A_inv (inverse covariance) and b (weighted sum of x*r)
        # We maintain A_inv and b, and compute theta = A_inv @ b when needed
        self.A_inv = np.eye(dim) * (1.0 / self.lam)
        self.b = np.zeros(dim)
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def _make_context(self, context, arm):
        """
        Create an augmented context vector by appending a one-hot encoding 
        of the arm to the original context. This allows a single global 
        linear model to capture arm-specific effects.
        """
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate([context, one_hot])

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: greedy based on current mean estimate
            scores = []
            for a in range(self.n):
                x_aug = self._make_context(context, a)
                theta = self.A_inv @ self.b
                scores.append(theta @ x_aug)
            return int(np.argmax(scores))

        # Thompson Sampling: sample theta from N(mu, A_inv)
        mu = self.A_inv @ self.b
        
        # Efficient sampling: theta = mu + L @ z, where L is Cholesky of A_inv
        # To avoid Cholesky every step (which is O(d^3)), we can use a 
        # precomputed or approximated structure, but for d=dim (small), 
        # Cholesky is fine. Alternatively, we can sample using the 
        # eigendecomposition if we cached it, but A_inv changes.
        # Let's just do Cholesky. It's O(d^3) but d is small (e.g., 5-10).
        try:
            L = np.linalg.cholesky(self.A_inv)
            z = np.random.randn(self.d)
            theta_sample = mu + L @ z
        except np.linalg.LinAlgError:
            # If A_inv is not positive definite (shouldn't happen with lam>0),
            # fall back to greedy
            scores = []
            for a in range(self.n):
                x_aug = self._make_context(context, a)
                scores.append(mu @ x_aug)
            return int(np.argmax(scores))

        # Select arm with highest sampled reward
        scores = []
        for a in range(self.n):
            x_aug = self._make_context(context, a)
            scores.append(theta_sample @ x_aug)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Check deadline
        if time.time() > self.deadline:
            # Minimal update
            x_aug = self._make_context(context, arm)
            self.A_inv = self.beta_forget * self.A_inv
            self.b = self.beta_forget * self.b
            # Sherman-Morrison-Woodbury update
            # A_new = beta*A_old + x x^T
            # A_new_inv = (1/beta) A_old_inv - (1/beta^2) (A_old_inv x x^T A_old_inv) / (beta + x^T A_old_inv x)
            # But we are scaling A_inv by beta, so let's do it carefully.
            # Actually, it's easier to just do the standard update without scaling first,
            # but we want forgetting.
            # Let's stick to the simple update for safety in deadline
            self.A_inv += np.outer(x_aug, x_aug)
            self.b += reward * x_aug
            return

        x_aug = self._make_context(context, arm)
        
        # Forget: scale A_inv and b
        self.A_inv = self.beta_forget * self.A_inv
        self.b = self.beta_forget * self.b
        
        # Add new observation using Sherman-Morrison-Woodbury
        # We want to update A_inv such that it corresponds to A_old + x x^T
        # But we scaled A_inv by beta, so it corresponds to beta * A_old_inv
        # Let's just do the standard rank-1 update on the scaled A_inv
        # A_new_inv = A_old_inv - (A_old_inv x x^T A_old_inv) / (1 + x^T A_old_inv x)
        # This is for A_new = A_old + x x^T
        # Since our A_inv is already scaled, this is correct.
        
        Ax = self.A_inv @ x_aug
        denom = 1.0 + x_aug @ Ax
        if denom > 1e-8:
            self.A_inv -= np.outer(Ax, Ax) / denom
        
        self.b += reward * x_aug
# EVOLVE-BLOCK-END
