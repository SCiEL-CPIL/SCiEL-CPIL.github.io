"""Optimized BBOB f14 solver with Gaussian Process Model-Based Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Gaussian Process MBO optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(123)
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        X_samples = []
        Y_samples = []

        # Helper to safely evaluate function and update best
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            X_samples.append(x.copy())
            Y_samples.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Random Sampling
        # We need enough points to build an initial GP model.
        # 10-15% of budget is reasonable for dim=5.
        n_init = max(10, int(budget * 0.15))
        
        # Ensure we don't waste budget if n_init > budget
        n_init = min(n_init, budget)
        
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # If we ran out of budget in init, return best
        if calls_made >= budget:
            if best_x is None:
                best_x = np.zeros(dim)
                best_f = func(best_x)
            return best_x, best_f

        # Phase 2: Gaussian Process Model-Based Optimization
        # We will use a simple GP implementation with RBF kernel.
        # To avoid dependency on sklearn (which might be slow or unavailable in some contexts, 
        # though the prompt says SciPy is available, standard lib is safer for GP math 
        # unless we use scipy.stats which doesn't have GP). 
        # Actually, the prompt says "Python standard library, NumPy and SciPy are available."
        # Implementing GP from scratch with NumPy is robust and fast for dim=5, N~500.

        # GP Parameters
        # RBF Kernel: k(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        # We can use fixed hyperparameters for simplicity and speed, 
        # or adapt them. Fixed is safer for budget constraints.
        # For BBOB f14 (Different powers), the landscape can be anisotropic.
        # A single length scale might be suboptimal, but let's try a reasonable default.
        # Box is [-5, 5]^5. Diameter is 10*sqrt(5) ~ 22.
        # Length scale l=1.0 to 2.0 is often good for these scales.
        l = 1.5
        sigma_f = 1.0
        noise = 1e-8

        def kernel(X1, X2):
            # X1: (N, D), X2: (M, D)
            # Returns (N, M) kernel matrix
            # Compute squared distances
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
            X1_sq = np.sum(X1**2, axis=1)
            X2_sq = np.sum(X2**2, axis=1)
            dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
            dist_sq = np.maximum(dist_sq, 0) # numerical stability
            return sigma_f**2 * np.exp(-0.5 * dist_sq / (l**2))

        # Main Loop
        # We continue until budget or time is exhausted.
        # Each iteration:
        # 1. Build GP model from X_samples, Y_samples
        # 2. Generate candidate points (e.g., grid or random)
        # 3. Compute Expected Improvement (EI)
        # 4. Select best candidate
        # 5. Evaluate and add to samples

        # To make EI computation efficient, we evaluate EI on a set of candidates.
        # A random set of candidates is sufficient for dim=5.
        n_candidates = 200
        
        while calls_made < budget and time.time() < deadline:
            X = np.array(X_samples)
            Y = np.array(Y_samples)
            
            # Normalize Y for GP stability? 
            # GP is invariant to linear transformations of Y if we transform back, 
            # but scaling Y helps with numerical precision.
            # Let's scale Y to have mean 0 and std 1 for the GP fit.
            y_mean = np.mean(Y)
            y_std = np.std(Y)
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y - y_mean) / y_std

            # Compute Kernel Matrix K
            K = kernel(X, X)
            K += noise * np.eye(len(X))
            
            # Solve for alpha: K * alpha = Y_norm
            # Use Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                # Fallback to direct solve if Cholesky fails
                try:
                    alpha = np.linalg.solve(K, Y_norm)
                except np.linalg.LinAlgError:
                    # If linear algebra fails, break and return best
                    break

            # Generate candidates
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Compute Kernel between X and X_cand
            K_s = kernel(X_cand, X) # (N_cand, N_samples)
            
            # Predictive Mean and Variance
            # mu(x) = k(x, X) * alpha
            mu = K_s @ alpha
            # var(x) = k(x, x) - k(x, X) * K^-1 * k(x, X)^T
            # k(x, x) is sigma_f^2 for RBF
            v = np.linalg.solve(L, K_s.T) # (N_samples, N_cand)
            var = sigma_f**2 - np.sum(K_s * v.T, axis=1) # (N_cand,)
            
            # Convert predictions back to original scale
            mu_orig = mu * y_std + y_mean
            var_orig = var * (y_std**2)
            std_orig = np.sqrt(np.maximum(var_orig, 0))

            # Compute Expected Improvement (EI)
            # EI = (f_best - mu) * Phi((f_best - mu) / std) + std * phi((f_best - mu) / std)
            # where f_best is the current best (minimum)
            # Note: EI is usually defined for maximization. For minimization, 
            # we maximize the improvement over the best (i.e., best - f(x)).
            # So we use f_best - mu.
            
            z = (best_f - mu_orig) / std_orig
            # Handle std=0 case
            z = np.where(std_orig > 1e-8, z, 0)
            
            # Standard Normal CDF and PDF
            # Using math.erf for CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
            # PDF: phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
            
            # Vectorized erf is not in standard numpy, but we can use scipy.stats or math.
            # Since we have an array, let's use a simple approximation or loop if small, 
            # but n_candidates=200 is small.
            # Actually, scipy is available.
            from scipy.stats import norm
            
            ei = (best_f - mu_orig) * norm.cdf(z) + std_orig * norm.pdf(z)
            
            # Select candidate with highest EI
            idx_best_cand = np.argmax(ei)
            x_next = X_cand[idx_best_cand]
            
            # Evaluate
            f_next = evaluate(x_next)
            
            # If evaluation failed or returned inf, we should probably not add it to GP
            # but evaluate() handles budget. If fx is inf, it's not added to X_samples?
            # Wait, my evaluate() adds to X_samples regardless of fx value if fx is not inf?
            # Actually, if fx is inf, it's a valid float but useless. 
            # BBOB functions usually return finite floats. 
            # If an exception occurs in func, evaluate catches it? No, evaluate doesn't catch exceptions from func.
            # The prompt says "an exception later keeps the calls made".
            # So if func raises, the program crashes. We assume func is well-behaved.
            
        # Ensure best_x is the absolute best found
        if best_x is None:
             best_x = np.zeros(dim)
             best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
