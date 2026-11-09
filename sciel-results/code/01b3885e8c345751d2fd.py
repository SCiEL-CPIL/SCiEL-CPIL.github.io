import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Multi-objective optimizer using Bayesian Optimization with a Gaussian Process surrogate."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Fixed GP hyperparameters for stability and speed
        self.kernel_length_scales = np.ones(dim) * 0.5
        self.kernel_variance = 1.0
        self.noise = 1e-6

    def _matern52_kernel(self, X, Y, l, v, noise):
        """
        Compute the Matern 5/2 kernel matrix between X and Y.
        X: (n, d), Y: (m, d)
        """
        # Euclidean distances
        X2 = np.sum(X**2, axis=1)[:, np.newaxis]
        Y2 = np.sum(Y**2, axis=1)[np.newaxis, :]
        D2 = np.maximum(X2 + Y2 - 2 * X @ Y.T, 0)
        D = np.sqrt(D2)
        
        # Matern 5/2
        sqrt5_D = np.sqrt(5) * D / l
        K = (1 + sqrt5_D + 5/3 * D**2 / l**2) * np.exp(-sqrt5_D) * v
        
        if X.shape[0] == Y.shape[0] and np.array_equal(X, Y):
            K += noise * np.eye(X.shape[0])
            
        return K

    def _fit_gp(self, X, Y):
        """
        Fit GP to data. Returns K_inv and K_inv_Y for fast prediction.
        X: (n, d), Y: (n, m)
        """
        n = X.shape[0]
        K = self._matern52_kernel(X, X, self.kernel_length_scales, self.kernel_variance, self.noise)
        
        # Add jitter for numerical stability
        K += 1e-8 * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            K_inv_L = np.linalg.solve(L, np.eye(n))
            K_inv = K_inv_L.T @ K_inv_L
            K_inv_Y = K_inv @ Y
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            K_inv = np.linalg.pinv(K)
            K_inv_Y = K_inv @ Y
            
        return K_inv, K_inv_Y, X

    def _predict_gp(self, X_new, K_inv, K_inv_Y, X_train):
        """
        Predict mean and variance for X_new.
        Returns mu: (n_new, m), sigma: (n_new, m)
        """
        K_star = self._matern52_kernel(X_new, X_train, self.kernel_length_scales, self.kernel_variance, 0.0)
        
        mu = K_star @ K_inv_Y
        # Variance: v - K_star @ K_inv @ K_star.T
        # Note: K_star is (n_new, n_train), K_inv is (n_train, n_train)
        K_star_K_inv = K_star @ K_inv
        var = self.kernel_variance - np.sum(K_star_K_inv * K_star.T, axis=1)
        var = np.maximum(var, 1e-12)
        sigma = np.sqrt(var)
        
        return mu, sigma

    def _compute_moei(self, mu, sigma, ref_values):
        """
        Compute Multi-Objective Expected Improvement.
        mu: (n, m), sigma: (n, m)
        ref_values: (m,) - best known value for each objective (min)
        """
        n = mu.shape[0]
        m = self.n_obj
        EI = np.zeros(n)
        
        for k in range(m):
            # We minimize, so improvement is ref - f
            # f ~ N(mu, sigma^2)
            # E[max(0, ref - f)]
            # Let Z = (f - mu) / sigma ~ N(0, 1)
            # ref - f = ref - mu - sigma * Z
            # Max(0, A - sigma*Z) where A = ref - mu
            # If A <= 0, EI = 0 (mostly, actually strictly 0 if we ignore prob of f < ref when mu > ref? No, if mu > ref, there is still prob f < ref)
            
            A = ref_values[k] - mu[:, k]
            s = sigma[:, k]
            
            # Standard EI formula for minimization
            # EI = A * Phi(A/s) + s * phi(A/s)
            # Where Phi is CDF, phi is PDF of standard normal
            
            Z = A / s
            EI_k = A * norm.cdf(Z) + s * norm.pdf(Z)
            
            # Ensure non-negative
            EI_k = np.maximum(EI_k, 0)
            EI += EI_k
            
        return EI

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial sampling: Latin Hypercube
        n_init = 20
        # Simple LHS
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            X_init[:, d] = (perm + 0.5) / n_init
        X_init = X_init * (hi - lo) + lo
        
        F_init = np.array([problem(x) for x in X_init])
        eval_count = n_init
        
        # Store all evaluated points for archive
        all_X = [X_init[i] for i in range(n_init)]
        all_F = [F_init[i] for i in range(n_init)]
        
        # Archive: non-dominated solutions
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any in archive
            for i in range(len(archive_F)):
                if np.all(archive_F[i] <= f) and np.any(archive_F[i] < f):
                    return # f is dominated, discard
            
            # Remove any archive members dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            archive_X.append(x)
            archive_F.append(f)

        # Initial archive update
        for i in range(n_init):
            update_archive(X_init[i], F_init[i])
            
        # Main BO Loop
        n_candidates = 1000
        
        while eval_count < self.budget:
            if time.time() > deadline - 5.0:
                break
                
            # Fit GP
            X_train = np.array(all_X)
            Y_train = np.array(all_F)
            
            # Ensure unique points for GP (GP fails with duplicate X)
            # If duplicates exist, average Y or just keep one. 
            # Since we pick new points, duplicates are unlikely unless random candidate hits exact previous point.
            # To be safe, we can check uniqueness.
            
            K_inv, K_inv_Y, X_train_unique = self._fit_gp(X_train, Y_train)
            
            # Generate candidates
            X_cand = np.random.uniform(lo, hi, size=(n_candidates, self.dim))
            
            # Predict
            mu, sigma = self._predict_gp(X_cand, K_inv, K_inv_Y, X_train)
            
            # Compute MOEI
            # Ref values: best known for each objective
            ref_values = np.array([np.min(archive_F, axis=0) if archive_F else np.max(hi)])
            
            EI = self._compute_moei(mu, sigma, ref_values)
            
            # Select best candidate
            best_idx = np.argmax(EI)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            eval_count += 1
            
            all_X.append(x_new)
            all_F.append(f_new)
            
            update_archive(x_new, f_new)

        return np.array(all_X)
