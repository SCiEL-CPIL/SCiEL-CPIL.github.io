"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Gaussian Process surrogate-based optimizer for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = np.random.uniform(-5, 5, dim)
        f_global_best = np.inf
        calls = 0
        
        # Storage for GP
        X = []
        Y = []
        
        # Kernel parameters (initial guesses)
        # Length scales: start with medium, let optimization adjust if we did, but we won't do full MLE to save time.
        # We will use a fixed, reasonable set of hyperparameters for the Matern 5/2 kernel.
        # f22 has peaks of varying sizes. A length scale of 1.0-2.0 is reasonable for 5D [-5,5].
        length_scales = np.ones(dim) * 1.5
        noise_var = 1e-8
        signal_var = 1.0
        
        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        def matern52_kernel(x1, x2, ls, sv):
            # x1: (n, d), x2: (m, d)
            # dist: (n, m)
            dist2 = np.sum(x1**2, axis=1)[:, np.newaxis] + np.sum(x2**2, axis=1)[np.newaxis, :] - 2 * x1 @ x2.T
            dist2 = np.maximum(dist2, 0.0)
            dist = np.sqrt(dist2)
            sqrt5_dist = np.sqrt(5.0) * dist / ls
            K = sv**2 * (1.0 + sqrt5_dist + 5.0/3.0 * dist2 / (ls**2)) * np.exp(-sqrt5_dist)
            return K

        # Phase 1: Latin Hypercube Sampling for initial coverage
        # LHS ensures spread across the box.
        n_init = 50
        # Generate LHS points
        x_lhs = np.zeros((n_init, dim))
        for i in range(n_init):
            for j in range(dim):
                x_lhs[i, j] = (i + rng.uniform()) / n_init
        x_lhs = x_lhs * 10.0 - 5.0
        
        for i in range(n_init):
            if not check_time() or calls >= budget:
                break
            evaluate(x_lhs[i])
            X.append(x_lhs[i].copy())
            Y.append(f_global_best if i==0 else func(x_lhs[i])) # Re-eval to get exact Y for GP? No, we stored f in evaluate.
            
        # Correctly populate X and Y from the loop above. 
        # The evaluate function updates global best, but we need the specific f for each x.
        # Let's re-implement the loop to store Y properly.
        X = []
        Y = []
        calls = 0
        f_global_best = np.inf
        x_global_best = np.zeros(dim)
        
        for i in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = x_lhs[i]
            f = evaluate(x)
            X.append(x.copy())
            Y.append(f)
            
        X = np.array(X)
        Y = np.array(Y)

        if calls >= budget or not check_time():
            return x_global_best, f_global_best

        # Phase 2: GP Active Learning
        # We will iterate until budget or time is up.
        # To speed up GP prediction, we limit the number of points used in the GP to the most recent or most diverse ones?
        # No, with 500 calls, GP on 500x500 matrix is fast enough in NumPy if we are careful.
        # However, Cholesky decomposition of 500x500 is ~ 500^3/3 = 40M ops. This is very fast (<0.1s).
        # So we can use all points.
        
        # We need to handle the case where the GP might suggest points very close to existing ones.
        # We add a small jitter to the diagonal of K to ensure positive definiteness.
        
        while check_time() and calls < budget:
            if len(X) < 2:
                # Fallback to random if GP fails or is too small
                x_next = rng.uniform(-5, 5, dim)
            else:
                # Build Kernel Matrix
                K = matern52_kernel(X, X, length_scales, signal_var)
                K += noise_var * np.eye(len(X))
                
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Add more jitter if singular
                    K += 1e-6 * np.eye(len(X))
                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        # Fallback to random
                        x_next = rng.uniform(-5, 5, dim)
                        evaluate(x_next)
                        X = np.vstack([X, x_next])
                        Y = np.append(Y, func(x_next))
                        continue

                # Mean and Variance prediction
                # We want to find x that maximizes EI.
                # EI(x) = (mu_f - f_best - xi) * Phi((mu_f - f_best - xi)/sigma_f) + sigma_f * phi((mu_f - f_best - xi)/sigma_f)
                # where xi is a small exploration parameter.
                
                # To find the max EI, we can sample candidate points from the prior or use a grid?
                # In 5D, a grid is too expensive. 
                # Strategy: Sample a batch of candidate points from a distribution centered on the current best,
                # evaluate their EI using the GP, and pick the best one.
                
                n_candidates = 100
                # Sample candidates from a Gaussian around the global best with a decaying sigma
                # Or sample from the prior (uniform)? 
                # Mixing: 50% around best, 50% random exploration?
                # Let's use a "batch" of candidates.
                
                candidates = []
                # 50 candidates near best
                sigma_explore = 1.0 # Fixed exploration scale for candidates
                for _ in range(n_candidates // 2):
                    candidates.append(x_global_best + rng.normal(0, sigma_explore, dim))
                # 50 candidates random
                for _ in range(n_candidates // 2):
                    candidates.append(rng.uniform(-5, 5, dim))
                
                candidates = np.array(candidates)
                candidates = np.clip(candidates, -5.0, 5.0)
                
                # Predict mean and var for candidates
                K_s = matern52_kernel(candidates, X, length_scales, signal_var) # (n_cand, n_data)
                
                # Solve for alpha
                # K X = y  =>  L L^T X = y
                # L v = y
                v = np.linalg.solve(L, Y)
                # L^T alpha = v
                alpha = np.linalg.solve(L.T, v)
                
                # Mean: K_s @ alpha
                mu_s = K_s @ alpha
                # Var: signal_var - K_s @ K_s.T
                # Efficient var calculation:
                # v_s = L^{-1} K_s^T
                # var = signal_var - sum(v_s^2, axis=0)
                v_s = np.linalg.solve(L, K_s.T)
                var_s = signal_var - np.sum(v_s**2, axis=0)
                var_s = np.maximum(var_s, 1e-8)
                sigma_s = np.sqrt(var_s)
                
                # Calculate EI
                f_best = np.min(Y)
                xi = 0.01 * (np.max(Y) - f_best) # Adaptive xi? Or fixed small.
                # Standard EI for minimization:
                # Improvement = f_best - f(x)
                # E[Improve] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
                
                z = (f_best - mu_s) / sigma_s
                EI = (f_best - mu_s) * norm.cdf(z) + sigma_s * norm.pdf(z)
                
                # Add a small random noise to break ties or encourage exploration if EI is flat
                EI += rng.normal(0, 1e-6, len(EI))
                
                idx_best = np.argmax(EI)
                x_next = candidates[idx_best]
                
            # Evaluate the next point
            f_next = evaluate(x_next)
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            # Optional: Adaptive length scales?
            # If we get stuck, we might want to shrink or grow length scales.
            # For simplicity and robustness, we keep them fixed.
            # However, if the GP is very confident (low var everywhere), we might be stuck in a local min.
            # The "random candidates" in the pool above helps escape.

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
