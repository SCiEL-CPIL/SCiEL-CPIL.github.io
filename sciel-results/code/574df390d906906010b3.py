import numpy as np
import time

class Policy:
    """
    Bayesian Linear Bandit with Thompson Sampling and Adaptive Forgetting.
    
    Uses a single linear model per arm.
    The posterior precision matrix A and cross-moment b are updated with an adaptive
    forgetting factor alpha.
    Alpha is determined by the standardized residual (Mahalanobis distance) of the
    latest observation. Large errors trigger fast forgetting (low alpha) to adapt
    to regime shifts, while small errors allow slow forgetting (high alpha) to
    accumulate data and reduce uncertainty.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize precision matrices (A) and cross-moments (b)
        # Start with a prior of N(0, I) -> A = I, b = 0
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.alpha_min = 0.90
        self.alpha_max = 0.999
        self.gamma = 1.0  # Scaling factor for the exponential decay of alpha
        
        # Numerical stability
        self.epsilon = 1e-6
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s margin from 120s limit

    def _get_theta_and_sample(self, A, b):
        """
        Computes the posterior mean theta_mean and samples a theta_sample 
        from the posterior distribution N(theta_mean, A^-1).
        """
        # Cholesky decomposition of A = L L^T
        try:
            L_A = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Fallback if A is not positive definite
            # Add small regularization
            A_reg = A + 1e-6 * np.eye(self.d)
            L_A = np.linalg.cholesky(A_reg)
            # Note: This changes the distribution slightly, but prevents crash

        # Calculate theta_mean = A^-1 b
        # Solve L_A y = b
        y = np.linalg.solve(L_A, b)
        # Solve L_A^T theta = y
        theta_mean = np.linalg.solve(L_A.T, y)
        
        # Sample u ~ N(0, A^-1)
        # We know A^-1 = L_A^-T L_A^-1
        # So u = L_A^-T z where z ~ N(0, I)
        z = np.random.randn(self.d)
        # Solve L_A^T u = z
        u = np.linalg.solve(L_A.T, z)
        
        theta_sample = theta_mean + u
        return theta_mean, theta_sample

    def select(self, context):
        if time.time() > self.deadline:
            return 0
            
        context = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            try:
                _, theta_sample = self._get_theta_and_sample(self.A[a], self.b[a])
                scores[a] = np.dot(context, theta_sample)
            except Exception:
                # If any error occurs, assign low score
                scores[a] = -1e9
                
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.array(context)
        
        try:
            # 1. Calculate prediction stats with current model
            # Get theta_mean and variance v = x^T A^-1 x
            # We can reuse the logic from _get_theta_and_sample but we also need v
            
            try:
                L_A = np.linalg.cholesky(self.A[arm])
            except np.linalg.LinAlgError:
                L_A = np.linalg.cholesky(self.A[arm] + 1e-6 * np.eye(self.d))

            # theta_mean
            y = np.linalg.solve(L_A, self.b[arm])
            theta_mean = np.linalg.solve(L_A.T, y)
            
            # Variance v = x^T A^-1 x
            # Solve A w = x
            # L_A y_v = x
            y_v = np.linalg.solve(L_A, context)
            # L_A^T w = y_v
            w = np.linalg.solve(L_A.T, y_v)
            v = np.dot(context, w)
            
            # 2. Calculate residual and standardized residual (Mahalanobis distance)
            pred = np.dot(context, theta_mean)
            residual = reward - pred
            z = residual / np.sqrt(v + self.epsilon)
            
            # 3. Determine adaptive alpha
            # Large |z| -> small alpha (fast forgetting)
            # Small |z| -> large alpha (slow forgetting)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-self.gamma * abs(z))
            
            # 4. Update A and b
            self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
            self.b[arm] = alpha * self.b[arm] + reward * context

        except Exception:
            # Fallback update with fixed alpha if something goes wrong
            alpha = 0.95
            self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
            self.b[arm] = alpha * self.b[arm] + reward * context
