import numpy as np

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A_inv (inverse of covariance) and b
        # A starts as I, so A_inv starts as I
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive parameters
        self.alpha_min = 0.9
        self.alpha_max = 0.999
        self.alphas = [0.99 for _ in range(n_arms)] # Initial discount
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.beta_ema = 0.1 # Rate for residual EMA update
        
        # Exploration parameter
        self.exploration_beta = 1.0
        
        # Safety/Time tracking
        self.t = 0
        self.start_time = np.datetime64('now')

    def select(self, context):
        # Time guard
        # If time is running out, we could switch to greedy or simple heuristic, 
        # but for 120s, we are likely safe with O(d^2) updates.
        
        scores = []
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mu = theta @ context
            # Variance term
            var = context @ self.A_inv[a] @ context
            # Ensure var is positive
            if var < 0: var = 0
            ucb = mu + self.exploration_beta * np.sqrt(var)
            scores.append(ucb)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # 1. Predict to get residual
        theta = self.A_inv[arm] @ self.b[arm]
        pred = theta @ context
        residual = reward - pred
        res_sq = residual * residual
        
        # 2. Update residual EMA
        self.residual_ema[arm] = (1 - self.beta_ema) * self.residual_ema[arm] + self.beta_ema * res_sq
        
        # 3. Adapt alpha
        # Map residual_ema to alpha. High error -> low alpha (forget fast).
        # Heuristic: alpha = alpha_max - (alpha_max - alpha_min) * min(1, residual_ema / threshold)
        # Let's use a smooth mapping.
        # If residual is 0, alpha = alpha_max.
        # If residual is large, alpha = alpha_min.
        # threshold = 1.0 (standard deviation assumption)
        r = self.residual_ema[arm]
        # Normalize r. If r > 1, we are in high non-stationarity.
        # alpha = alpha_max - (alpha_max - alpha_min) * (r / (r + 1.0))
        # This ensures alpha is in [alpha_min, alpha_max].
        alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * (r / (r + 1.0))
        self.alphas[arm] = alpha
        
        # 4. Update A_inv using Woodbury
        # A_new = alpha * A_old + x x^T
        # A_new^{-1} = 1/alpha * A_old^{-1} - (1/alpha^2) * A_old^{-1} x x^T A_old^{-1} / (1 + 1/alpha * x^T A_old^{-1} x)
        
        A_inv = self.A_inv[arm]
        x = context
        alpha = self.alphas[arm]
        
        # Compute intermediate terms
        Vx = A_inv @ x  # d x 1
        xVx = x @ Vx    # scalar
        denom = 1.0 + (1.0 / alpha) * xVx
        
        # Woodbury update
        # Term 1: (1/alpha) * A_inv
        # Term 2: (1/alpha^2) * Vx x^T A_inv / denom
        # A_inv_new = (1/alpha)*A_inv - (1/alpha^2)*np.outer(Vx, x) @ A_inv / denom
        # Note: np.outer(Vx, x) is dxd. @ A_inv is dxd.
        # More efficient:
        # A_inv_new = (1/alpha)*A_inv - (1/(alpha*denom)) * np.outer(Vx, (1/alpha)*A_inv @ x) ?? No.
        # Let's stick to the formula:
        # A_inv_new = (1/alpha) * A_inv - (1/(alpha**2 * denom)) * np.outer(Vx, Vx) * (x^T x? No)
        
        # Correct Woodbury for (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = alpha * A_old, u = x, v = x.
        # (alpha A_old + x x^T)^{-1} = (alpha A_old)^{-1} - (alpha A_old)^{-1} x x^T (alpha A_old)^{-1} / (1 + x^T (alpha A_old)^{-1} x)
        # = (1/alpha) A_old^{-1} - (1/alpha^2) A_old^{-1} x x^T A_old^{-1} / (1 + (1/alpha) x^T A_old^{-1} x)
        
        # Let M = A_old^{-1}
        # M_new = (1/alpha) M - (1/(alpha^2 * denom)) * (M x) (x^T M)
        # Since M is symmetric, M x = Vx, x^T M = Vx^T.
        # M_new = (1/alpha) M - (1/(alpha^2 * denom)) * np.outer(Vx, Vx)
        
        M = A_inv
        M_new = (1.0 / alpha) * M
        outer_Vx = np.outer(Vx, Vx)
        M_new -= (1.0 / (alpha * alpha * denom)) * outer_Vx
        
        # Symmetrize to prevent numerical drift
        M_new = (M_new + M_new.T) / 2.0
        self.A_inv[arm] = M_new
        
        # 5. Update b
        self.b[arm] = alpha * self.b[arm] + reward * x
