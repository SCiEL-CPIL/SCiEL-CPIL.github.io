"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Data storage
        X = []
        Y = []
        
        best_x = None
        best_f = float('inf')
        total_calls = 0

        def add_point(x, y):
            nonlocal best_x, best_f
            X.append(x)
            Y.append(y)
            if y < best_f:
                best_f = y
                best_x = x.copy()

        # Phase 1: Initial Random Sampling
        # Use 10% of budget for initial exploration to seed the GP
        n_init = max(10, budget // 10)
        
        # Latin Hypercube Sampling for better coverage
        X_init = np.zeros((n_init, n))
        for j in range(n):
            X_init[:, j] = (np.arange(n_init) + rng.uniform(size=n_init)) / n_init
            rng.shuffle(X_init[:, j])
        X_init = X_init * 10 - 5 # Scale to [-5, 5]

        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline:
                break
            x = X_init[i]
            y = func(x)
            total_calls += 1
            add_point(x, y)

        # Phase 2: Bayesian Optimization Loop
        # GP Parameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6
        
        # Acquisition: Expected Improvement (EI)
        def kernel(X1, X2):
            """Matern 5/2 kernel, good for non-smooth functions like Katsuura"""
            # Compute pairwise distances
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            dists = np.sqrt(sq_dists)
            z = dists / length_scale
            return signal_var * (1 + z + z**2 / 3.0) * np.exp(-z)

        def fit_gp(X_train, Y_train):
            """Fit GP hyperparameters by maximizing log-likelihood (simplified: grid search or fixed)"""
            # For speed and stability, we will use fixed hyperparameters tuned for the box [-5,5]^5
            # and just compute the predictive distribution.
            # In a real BO loop, one might optimize length_scale and signal_var.
            # Given the 300s limit, we'll stick to a robust fixed setup or very light adaptation.
            
            # Let's try to adapt length_scale slightly based on the spread of best points
            # But for now, fixed is safer to avoid GP fitting overhead.
            K = kernel(X_train, X_train)
            K += noise_var * np.eye(len(X_train))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Regularize if not positive definite
                K += 1e-4 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return None, None, None
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            return K, L, alpha

        def predict_gp(X_pred, K, L, alpha):
            """Predict mean and variance at X_pred"""
            K_s = kernel(X_pred, X_train)
            v = np.linalg.solve(L, K_s.T)
            mu = K_s @ alpha
            var = signal_var - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def acquisition_ei(x, mu, var, best_f_current):
            """Expected Improvement acquisition function"""
            sigma = np.sqrt(var)
            if sigma < 1e-10:
                return 0.0
            z = (best_f_current - mu) / sigma
            ei = (best_f_current - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Main BO Loop
        while total_calls < budget and time.time() < deadline:
            # Fit GP
            X_train = np.array(X)
            Y_train = np.array(Y)
            
            # Normalize Y for GP stability? GP is invariant to affine transforms of Y if we transform back.
            # But kernel depends on X. Let's just use raw Y.
            # To handle scale, let's center Y
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train) + 1e-10
            Y_norm = (Y_train - Y_mean) / Y_std
            
            K, L, alpha = fit_gp(X_train, Y_norm)
            if K is None:
                # GP failed, fallback to random search for remaining budget
                for _ in range(budget - total_calls):
                    if total_calls >= budget or time.time() > deadline:
                        break
                    x_new = rng.uniform(-5, 5, n)
                    y_new = func(x_new)
                    total_calls += 1
                    add_point(x_new, y_new)
                break

            # Generate candidates for acquisition
            # Use a mix of random points and local perturbations around best points
            n_candidates = 200
            
            # 1. Random points
            X_rand = rng.uniform(-5, 5, (n_candidates // 2, n))
            
            # 2. Local perturbations around best points
            # Find top 5 points
            top_indices = np.argsort(Y_train)[:5]
            X_local = []
            for idx in top_indices:
                x_best = X_train[idx]
                # Perturb with decreasing sigma
                sigmas = [0.5, 1.0, 2.0]
                for s in sigmas:
                    x_pert = x_best + rng.normal(0, s, n)
                    x_pert = np.clip(x_pert, -5, 5)
                    X_local.append(x_pert)
            
            X_cand = np.vstack([X_rand, np.array(X_local)])
            
            # Predict on candidates
            mu, var = predict_gp(X_cand, K, L, alpha)
            
            # Calculate EI
            # Note: Y was normalized, so we need to be careful with best_f_current
            # best_f_current is in original scale.
            # The GP predicts normalized Y.
            # EI in normalized space: EI_norm = (best_norm - mu_norm) * Phi(z) + sigma_norm * phi(z)
            # Since we want to maximize improvement in original space, and the transform is linear,
            # the ranking is preserved.
            best_norm = (best_f - Y_mean) / Y_std
            z = (best_norm - mu) / np.sqrt(var)
            ei = (best_norm - mu) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            # Select best candidate
            idx_best = np.argmax(ei)
            x_new = X_cand[idx_best]
            
            # Evaluate
            y_new = func(x_new)
            total_calls += 1
            add_point(x_new, y_new)

        return best_x, best_f
# EVOLVE-BLOCK-END
