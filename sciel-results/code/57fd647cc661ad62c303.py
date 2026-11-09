"""Random Restart + Adaptive Local Search + Bayesian Optimization for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

class Optimizer:
    """
    Optimizes Katsuura f23 using:
    1. Phase 1: Random restarts with adaptive local refinement (Nelder-Mead-like).
    2. Phase 2: Bayesian Optimization with adaptive GP for fine-tuning.
    3. Deadline guard ensures we return best result before timeout.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Phase 1: Random Restart with Adaptive Local Refinement ---
        # Use budget allocation: 70% for global exploration, 30% for BO
        phase1_budget = int(budget * 0.7)
        
        # Track all evaluated points for BO initialization
        all_x = []
        all_f = []
        
        # Number of restarts: start with more, decrease as we get closer to optimum
        n_restarts = 20
        calls_per_restart = max(10, phase1_budget // n_restarts)
        
        for restart in range(n_restarts):
            if calls_made >= phase1_budget or check_deadline():
                break
                
            # Start point: random in the box, but bias toward center as restarts increase
            if restart == 0:
                x_start = rng.uniform(lower, upper, size=dim)
            else:
                # Mix of random and perturbed best
                if rng.random() < 0.3 and x_global_best is not None:
                    # Perturb the current best
                    sigma_perturb = 1.0 * (1 - restart / n_restarts) + 0.2
                    x_start = x_global_best + rng.normal(0, sigma_perturb, size=dim)
                    x_start = np.clip(x_start, lower, upper)
                else:
                    # Random point, but increasingly concentrated
                    concentration = 0.5 + 0.5 * (restart / n_restarts)
                    x_start = rng.uniform(lower + (upper-lower)*(1-concentration)/2, 
                                          upper - (upper-lower)*(1-concentration)/2, size=dim)
            
            x_curr, f_curr = evaluate(x_start)
            all_x.append(x_curr.copy())
            all_f.append(f_curr)
            
            # Local refinement: simple coordinate descent with adaptive step size
            step_size = 0.5
            min_step = 1e-4
            max_iter_local = calls_per_restart - 1
            local_iter = 0
            
            while local_iter < max_iter_local and calls_made < phase1_budget and not check_deadline():
                local_iter += 1
                
                # Try each coordinate with positive and negative steps
                best_delta = None
                best_f_delta = f_curr
                best_x_delta = None
                
                for d in range(dim):
                    for sign in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[d] += sign * step_size
                        x_trial = np.clip(x_trial, lower, upper)
                        
                        x_trial, f_trial = evaluate(x_trial)
                        all_x.append(x_trial.copy())
                        all_f.append(f_trial)
                        
                        if f_trial < best_f_delta - 1e-10:
                            best_f_delta = f_trial
                            best_delta = sign * step_size
                            best_x_delta = x_trial.copy()
                        
                        if calls_made >= phase1_budget or check_deadline():
                            break
                    if calls_made >= phase1_budget or check_deadline():
                        break
                
                if best_x_delta is not None:
                    # Move to the best position found
                    x_curr = best_x_delta
                    f_curr = best_f_delta
                    step_size = min(step_size * 1.2, 2.0)  # Increase step if improving
                else:
                    # No improvement: reduce step size
                    step_size *= 0.5
                    if step_size < min_step:
                        break
        
        # --- Phase 2: Bayesian Optimization for fine-tuning ---
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and not check_deadline() and len(all_x) >= 5:
            # Prepare training data
            X_train = np.array(all_x)
            y_train = np.array(all_f)
            
            # Adaptive GP: start with larger length scale, adapt based on improvement
            length_scale = 1.0
            noise = 1e-6
            signal_var = max(np.var(y_train), 1e-8)
            
            def rbf_kernel(X1, X2, ls, sv):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return sv * np.exp(-0.5 * sq_dists / ls**2)
            
            def gp_predict(X_test, X_tr, y_tr, ls, noise, sv):
                K = rbf_kernel(X_tr, X_tr, ls, sv) + noise * np.eye(len(X_tr))
                Ks = rbf_kernel(X_tr, X_test, ls, sv)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                    mu = Ks.T @ alpha
                    v = np.linalg.solve(L, Ks)
                    var = sv - np.sum(v**2, axis=0)
                    var = np.maximum(var, 1e-10)
                except np.linalg.LinAlgError:
                    mu = np.full(X_test.shape[0], np.mean(y_tr))
                    var = np.full(X_test.shape[0], np.var(y_tr) + 1e-10)
                return mu, var
            
            def expected_improvement(mu, var, f_best, xi=0.01):
                sigma = np.sqrt(var)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (f_best - mu - xi) / sigma
                    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                ei = np.where(sigma > 1e-10, ei, 0.0)
                return ei
            
            # Main BO loop with adaptive length scale
            bo_step = 0
            no_improve_count = 0
            f_best_prev = f_global_best
            last_ls_change = 0
            
            while calls_made < budget and not check_deadline():
                bo_step += 1
                
                # Adaptive exploration
                if bo_step < 5:
                    xi = 0.05
                elif bo_step < 15:
                    xi = 0.02
                else:
                    xi = 0.005
                
                # Adaptive length scale: shrink when improving, expand when stuck
                if no_improve_count > 5 and bo_step - last_ls_change > 10:
                    length_scale = min(length_scale * 1.5, 5.0)
                    last_ls_change = bo_step
                elif no_improve_count == 0 and bo_step - last_ls_change > 10:
                    length_scale = max(length_scale * 0.7, 0.1)
                    last_ls_change = bo_step
                
                # Generate candidates: mix of random and near-best
                n_candidates = 200
                X_candidates = rng.uniform(lower, upper, size=(n_candidates, dim))
                
                if x_global_best is not None:
                    n_near = 80
                    # Adaptive radius for near-best points
                    radius = 0.5 * (1 + 10 / (1 + bo_step))
                    X_near = x_global_best + rng.normal(0, radius, size=(n_near, dim))
                    X_near = np.clip(X_near, lower, upper)
                    X_candidates = np.vstack([X_candidates, X_near])
                
                # Compute GP predictions
                mu, var = gp_predict(X_candidates, X_train, y_train, length_scale, noise, signal_var)
                
                # Compute EI
                ei = expected_improvement(mu, var, f_global_best, xi)
                
                # Select best candidate
                best_idx = np.argmax(ei)
                x_next = X_candidates[best_idx]
                x_next, f_next = evaluate(x_next)
                
                # Update training set
                X_train = np.vstack([X_train, x_next.reshape(1, -1)])
                y_train = np.append(y_train, f_next)
                
                # Track improvement
                if f_next < f_best_prev - 1e-8:
                    no_improve_count = 0
                    f_best_prev = f_next
                else:
                    no_improve_count += 1
                
                # Periodic reseeding
                if no_improve_count > 12 and calls_made < budget - 5:
                    n_reseed = 2
                    for _ in range(n_reseed):
                        if calls_made >= budget or check_deadline():
                            break
                        x_rand = rng.uniform(lower, upper, size=dim)
                        x_rand, f_rand = evaluate(x_rand)
                        X_train = np.vstack([X_train, x_rand.reshape(1, -1)])
                        y_train = np.append(y_train, f_rand)
                    no_improve_count = 0

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
