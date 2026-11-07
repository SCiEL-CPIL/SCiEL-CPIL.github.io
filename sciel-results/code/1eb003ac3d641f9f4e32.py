# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Bayesian Optimization (GP + EI) with Local L-BFGS-B Refinement for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.perf_counter()

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            update_best(x, f)
            return f

        rng = np.random.default_rng(42)

        # Data storage for GP
        X_obs = []
        Y_obs = []

        def add_obs(x, f):
            X_obs.append(x.copy())
            Y_obs.append(f)

        # 1. Initialization: Latin Hypercube Sampling (LHS)
        n_init = 20
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            X_init[:, i] = np.sort(rng.uniform(-5.0, 5.0, n_init))
        
        # Evaluate initial points
        for i in range(n_init):
            if time_left() < 5.0 or calls_made >= budget:
                break
            f_val = wrapped_func(X_init[i])
            add_obs(X_init[i], f_val)

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)

        # GP Parameters
        # Matern 5/2 or RBF? RBF is smoother. f14 is smooth.
        # Let's use RBF (Squared Exponential)
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6

        def kernel(x1, x2):
            # x1: (n1, dim), x2: (n2, dim)
            # dist^2 = ||x1||^2 + ||x2||^2 - 2 x1.x2^T
            x1_sq = np.sum(x1**2, axis=1)
            x2_sq = np.sum(x2**2, axis=1)
            dist_sq = x1_sq[:, None] + x2_sq[None, :] - 2 * np.dot(x1, x2.T)
            dist_sq = np.maximum(dist_sq, 0)
            return signal_var * np.exp(-0.5 * dist_sq / (length_scale**2))

        def gp_predict(X_new):
            # X_new: (n_new, dim)
            K_xx = kernel(X_obs, X_obs)
            K_xx += noise_var * np.eye(len(X_obs))
            
            K_x_new = kernel(X_obs, X_new)
            K_new_new = kernel(X_new, X_new)
            
            # Solve K_xx alpha = Y_obs
            try:
                alpha = np.linalg.solve(K_xx, Y_obs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_xx, Y_obs, rcond=None)[0]
            
            mean = K_x_new.T @ alpha
            var = np.diag(K_new_new) - np.sum(K_x_new.T @ np.linalg.solve(K_xx, K_x_new), axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            return mean, std

        def expected_improvement(x_new, mean, std, best_y):
            # best_y is the minimum value found so far (we are minimizing)
            # EI = (best_y - mean) * Phi(z) + std * phi(z)
            # where z = (best_y - mean) / std
            # Note: Standard EI is for maximization. For minimization:
            # Improvement = best_y - f(x)
            # E[Improvement] = (best_y - mean) * Phi(z) + std * phi(z)
            # where z = (best_y - mean) / std
            z = (best_y - mean) / (std + 1e-10)
            # Use standard normal CDF and PDF
            from scipy.stats import norm
            phi = norm.pdf(z)
            Phi = norm.cdf(z)
            ei = (best_y - mean) * Phi + std * phi
            return ei

        # Main Loop
        while time_left() > 10.0 and calls_made < budget - 10:
            # 1. Fit GP and find next point
            # Use a grid of candidate points to maximize EI
            # In 5D, a full grid is too large. Use random sampling or LHS for candidates.
            n_candidates = 200
            X_cand = rng.uniform(-5.0, 5.0, (n_candidates, dim))
            
            # Also include the current best point's neighborhood?
            # Let's just use random candidates for global search
            
            means, stds = gp_predict(X_cand)
            
            current_best_y = min(Y_obs)
            
            eis = expected_improvement(X_cand, means, stds, current_best_y)
            
            # Select the candidate with max EI
            best_idx = np.argmax(eis)
            x_next = X_cand[best_idx]
            
            # 2. Local Refinement
            # Run L-BFGS-B from x_next
            bounds = [(-5.0, 5.0)] * dim
            try:
                # Limit maxfun to avoid wasting budget
                maxfun_local = min(50, max(10, (budget - calls_made) // 2))
                res = minimize(
                    wrapped_func,
                    x_next,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'maxls': 10,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxfun': maxfun_local
                    }
                )
                # The local optimizer has already updated best_x and best_f via wrapped_func
                # We need to add the final point of the local search to the GP data
                # res.x is the best point found by local search
                # But we need its function value.
                # wrapped_func updated best_f, but we need the specific value for x_obs
                # Let's re-evaluate or store it. 
                # Actually, wrapped_func updates global best. 
                # To add to GP, we need the (x, f) pair.
                # The last call in minimize might not be the best if it converged.
                # Let's just evaluate res.x once more to be sure? 
                # No, that wastes a call.
                # We can assume the last point in the local search history is good enough, 
                # or we can just add x_next and the result of the local search.
                # For simplicity, let's add the result of the local search.
                # We need to get the function value of res.x.
                # Since wrapped_func caches nothing, we must call it.
                # But we are close to budget.
                # Let's just add the starting point x_next and its EI-evaluated value?
                # No, x_next was not evaluated by wrapped_func yet, only by GP.
                # Wait, the loop is:
                # 1. GP suggests x_next.
                # 2. We run local search starting at x_next.
                # 3. Local search calls wrapped_func multiple times.
                # 4. The best point found by local search is now in best_x/best_f.
                # 5. We should add this best point to X_obs/Y_obs for the next GP iteration.
                
                # How to get the f value of res.x?
                # We can call wrapped_func(res.x) again? That costs 1 call.
                # Or we can track it. 
                # Let's just call wrapped_func(res.x) to register it properly in X_obs.
                # This is 1 extra call per iteration. With 500 budget, this is acceptable.
                
                f_res = wrapped_func(res.x)
                add_obs(res.x, f_res)
                
            except Exception:
                # If local search fails, just evaluate x_next directly
                f_next = wrapped_func(x_next)
                add_obs(x_next, f_next)

            # Update X_obs and Y_obs arrays for next GP fit
            if len(X_obs) > 0:
                X_obs = np.array(X_obs)
                Y_obs = np.array(Y_obs)
            else:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
