import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        
        # Helper: Evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Initial random sampling
        n_init = 30
        rng = np.random.default_rng(42)
        
        for i in range(n_init):
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        # Phase 2: Surrogate-assisted search with Gaussian Process
        # We'll use a simple GP with squared exponential kernel
        # and optimize Expected Improvement (EI) acquisition
        
        if eval_count < self.budget - 20:
            try:
                # Collect initial observations
                X_obs = [best_x.copy()]
                f_obs = [best_f]
                
                # Add a few more diverse points
                for i in range(10):
                    if eval_count >= self.budget or time.time() > deadline - 10:
                        break
                    x = rng.uniform(-5, 5, size=dim)
                    fx = evaluate(x)
                    X_obs.append(x.copy())
                    f_obs.append(fx)
                
                # GP hyperparameters
                length_scale = 1.0
                noise = 1e-6
                signal_var = 1.0
                
                def gp_predict(X_new, X_train, f_train):
                    """Simple GP prediction with squared exponential kernel"""
                    # Kernel matrix
                    K = np.zeros((len(X_train), len(X_train)))
                    for i in range(len(X_train)):
                        for j in range(len(X_train)):
                            d2 = np.sum((X_train[i] - X_train[j]) ** 2)
                            K[i, j] = signal_var * np.exp(-d2 / (2 * length_scale ** 2))
                    K += noise * np.eye(len(X_train))
                    
                    # Predict for new points
                    K_new = np.zeros((len(X_new), len(X_train)))
                    for i in range(len(X_new)):
                        for j in range(len(X_train)):
                            d2 = np.sum((X_new[i] - X_train[j]) ** 2)
                            K_new[i, j] = signal_var * np.exp(-d2 / (2 * length_scale ** 2))
                    
                    # GP predictions
                    K_inv = np.linalg.inv(K)
                    mu = K_new @ K_inv @ f_train
                    var = signal_var - K_new @ K_inv @ K_new.T
                    var = np.maximum(var, 1e-10)
                    std = np.sqrt(var)
                    
                    return mu, std
                
                def expected_improvement(mu, std, f_best, xi=0.01):
                    """Expected Improvement acquisition function"""
                    improvement = f_best - mu - xi
                    z = improvement / std
                    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                    return ei
                
                # Surrogate-assisted loop
                n_surrogate = 100
                for iter_idx in range(n_surrogate):
                    if eval_count >= self.budget or time.time() > deadline - 10:
                        break
                    
                    # Sample candidate points
                    n_candidates = 200
                    X_candidates = rng.uniform(-5, 5, size=(n_candidates, dim))
                    
                    # Predict with GP
                    X_obs_arr = np.array(X_obs)
                    f_obs_arr = np.array(f_obs)
                    mu, std = gp_predict(X_candidates, X_obs_arr, f_obs_arr)
                    
                    # Compute EI
                    ei = expected_improvement(mu, std, best_f)
                    
                    # Select best candidate
                    idx = np.argmax(ei)
                    x_new = X_candidates[idx]
                    
                    # Evaluate
                    fx_new = evaluate(x_new)
                    
                    # Update observations
                    X_obs.append(x_new.copy())
                    f_obs.append(fx_new)
                    
                    # Occasionally adapt length_scale based on improvement
                    if iter_idx % 20 == 19 and best_f < float('inf'):
                        # Simplified adaptation: reduce length_scale if we're making progress
                        length_scale *= 0.9
                        length_scale = max(length_scale, 0.1)
                        
            except Exception:
                pass
        
        # Phase 3: Local polish with (1+1)-ES
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            no_improve = 0
            limit = 50
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.05
                    no_improve = 0
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-5:
                        break
        
        return best_x, best_f
