<<<<<<< SEARCH
        # 3. Numerical Regularization
        # Symmetrize and add small regularization to prevent drift
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
        self.A_inv[arm] += self.reg * np.eye(self.d)
=======
        # 3. Numerical Regularization and Cholesky Update
        # Symmetrize and add small regularization to prevent drift
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
        self.A_inv[arm] += self.reg * np.eye(self.d)
        
        # Update Cholesky factor for next select() call
        try:
            self.L[arm] = np.linalg.cholesky(self.A_inv[arm])
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            eigvals, eigvecs = np.linalg.eigh(self.A_inv[arm])
            eigvals = np.maximum(eigvals, self.reg)
            self.L[arm] = eigvecs @ np.diag(np.sqrt(eigvals))
>>>>>>> REPLACE
