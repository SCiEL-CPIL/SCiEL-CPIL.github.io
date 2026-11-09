"""Gaussian Process Surrogate-based Optimizer with Local Polishing for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # State variables
        best_x = None
        best_f = np.inf
        n_calls = 0
        X = []  # List of evaluated points
        Y = []  # List of evaluated values

        rng = np.random.default_rng(42)

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Gaussian Process Surrogate ---
        
        # Matern 5/2 kernel
        def matern_kernel(X1, X2, length_scale, variance):
            # X1: (n1, d), X2: (n2, d)
            # Returns (n1, n2) kernel matrix
            # To avoid memory blowup for large N, we might need to be careful, 
            # but N<=500 is fine.
            d = dim
            # Compute squared distances
            # Use broadcasting: X1[:, None, :] - X2[None, :, :]
            # This creates (n1, n2, d) array. For 500x500x5, this is 1.25M floats, fine.
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist = np.sqrt(dist_sq + 1e-12)
            
            r = dist / length_scale
            K = variance * (1 + r + r**2 / 3) * np.exp(-r)
            return K

        # Initialize GP parameters
        # We will use a fixed kernel structure but adapt length scale slightly if needed?
        # For simplicity and robustness in 500 calls, we'll use a fixed reasonable length scale
        # or simple adaptive heuristic.
        # Initial length scale: ~1.0 to 2.0 seems reasonable for [-5,5] box.
        # Variance: 1.0
        # Noise: 1e-6
        
        def predict_gp(X_new, X_train, Y_train, length_scale, variance, noise):
            """
            Predict mean and variance for X_new given training data.
            X_new: (n_new, d)
            X_train: (n_train, d)
            Y_train: (n_train,)
            """
            n_train = len(Y_train)
            if n_train == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new)) * variance

            # Normalize Y to have mean 0 for numerical stability? 
            # Standard GP assumes Y ~ N(0, K).
            # Let's center Y.
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_centered = (Y_train - y_mean) / y_std

            K = matern_kernel(X_train, X_train, length_scale, 1.0) # Variance 1 for centered
            K += noise * np.eye(n_train)
            
            K_x = matern_kernel(X_train, X_new, length_scale, 1.0)
            
            try:
                # Cholesky decomposition for stability
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_centered))
                
                mean = y_mean + K_x.T @ alpha * y_std
                var = 1.0 - np.sum(L @ (np.linalg.solve(L, K_x)), axis=0)
                var = np.maximum(var, 1e-10)
                var = var * y_std**2
            except np.linalg.LinAlgError:
                # Fallback to pinv if Cholesky fails
                try:
                    alpha = np.linalg.pinv(K) @ Y_centered
                    mean = y_mean + K_x.T @ alpha * y_std
                    # Approximate variance
                    var = np.ones(len(X_new)) * y_std**2
                except Exception:
                    mean = np.full(len(X_new), y_mean)
                    var = np.full(len(X_new), y_std**2)

            return mean, var

        def expected_improvement(X_cand, current_best, length_scale, variance, noise, xi=0.01):
            """
            Calculate EI for candidate points.
            X_cand: (n_cand, d)
            """
            n_cand = len(X_cand)
            if len(Y) == 0:
                return np.zeros(n_cand)
                
            mu, sigma = predict_gp(X_cand, np.array(X), np.array(Y), length_scale, variance, noise)
            sigma = np.sqrt(sigma)
            
            # Avoid division by zero
            sigma[sigma < 1e-8] = 1e-8
            
            Z = (current_best - mu) / sigma
            EI = (current_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            # Add small exploration bonus if xi > 0? 
            # Standard EI handles this.
            return EI

        def local_polish(x_start, f_start, max_calls):
            """
            (1+1)-ES or Nelder-Mead style local refinement.
            """
            nonlocal n_calls, best_x, best_f
            x = x_start.copy()
            f = f_start
            sigma = 0.1
            d = dim
            
            # Simple (1+1)-ES with adaptive step size
            # We use this to clean up the result from GP.
            for _ in range(max_calls):
                if n_calls >= budget or check_time():
                    break
                
                # Generate neighbor
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                
                fy, y_eval = call_func(y)
                if fy is None:
                    break
                
                if fy < f:
                    x = y_eval
                    f = fy
                    sigma *= 1.05 # Slight increase
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.9 # Decrease
                    if sigma < 1e-6:
                        break
                        
            return x, f

        # --- Main Loop ---
        
        # Phase 1: Space-filling initialization
        # Use Latin Hypercube Sampling or Sobol
        # Sobol is better for low dimensions.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            init_n = min(30, budget // 2)
            init_points = sampler.random(init_n) * 10 - 5
        except ImportError:
            # Fallback to random if scipy not available (though it is)
            init_n = min(30, budget // 2)
            init_points = rng.uniform(-5, 5, size=(init_n, dim))
        
        for i in range(init_n):
            if n_calls >= budget or check_time():
                break
            x = init_points[i]
            fx, x_eval = call_func(x)
            if fx is None:
                continue
            X.append(x_eval)
            Y.append(fx)
            update_best(x_eval, fx)
        
        # Phase 2: GP-guided search
        # Parameters
        # Length scale: Start with 1.5. 
        # We can adapt this slightly, but fixed is robust for 500 calls.
        length_scale = 1.5
        variance = 1.0
        noise = 1e-5
        
        # Number of candidates to evaluate for EI at each step
        n_candidates = 100
        
        # Transition to polishing
        # If we have used 70% of budget, or if best hasn't improved in last 20 calls, polish.
        polish_threshold = int(budget * 0.7)
        last_improve_call = n_calls
        stagnation_limit = 20
        
        while n_calls < budget and not check_time():
            # Check for stagnation
            if n_calls - last_improve_call > stagnation_limit and n_calls > polish_threshold:
                break
            
            # Generate candidates
            # Sample from a box around the current best? Or full box?
            # Full box is safer for global exploration early on.
            # As we progress, we can shrink the box.
            # Let's sample from full box but weight towards best?
            # No, just sample uniformly from full box. EI will pick the best.
            
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Calculate EI
            ei_vals = expected_improvement(X_cand, best_f, length_scale, variance, noise)
            
            # Select best candidate
            best_idx = np.argmax(ei_vals)
            x_next = X_cand[best_idx]
            
            # Evaluate
            fx, x_eval = call_func(x_next)
            if fx is None:
                break
            
            X.append(x_eval)
            Y.append(fx)
            
            if fx < best_f:
                update_best(x_eval, fx)
                last_improve_call = n_calls
                # If we found a new best, we might want to reset stagnation counter
            else:
                # If no improvement, we might consider adapting length scale?
                # Keep it simple for now.
                pass
            
            # Periodically check time and budget
            if n_calls >= budget:
                break

        # Phase 3: Local Polishing
        if n_calls < budget and not check_time() and best_x is not None:
            remaining = budget - n_calls
            # Use at least 20% of remaining budget for polishing, but cap it
            polish_budget = max(20, int(remaining * 0.5))
            local_polish(best_x, best_f, polish_budget)

        return best_x, best_f

# EVOLVE-BLOCK-END
