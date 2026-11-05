<<<<<<< SEARCH
    def update(self, context, arm, reward):
        ctx = np.array(context, dtype=np.float64)
        # Update A and b with exponential forgetting
        self.A = self.gamma * self.A + np.outer(ctx, ctx)
        self.b[arm] = self.gamma * self.b[arm] + ctx * reward
        # Update inverse covariance using Sherman-Morrison or direct inversion
        # For small dim, direct inversion is fine
        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.eye(self.dim) * 0.1
        # Update theta
        self.theta[arm] = self.A_inv @ self.b[arm]
=======
    def update(self, context, arm, reward):
        # Update EMA for the selected arm
        self.ema[arm] = self.gamma * self.ema[arm] + (1 - self.gamma) * reward
>>>>>>> REPLACE
