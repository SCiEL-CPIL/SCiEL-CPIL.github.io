import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 135.0  # Leave 15s buffer for safety (evaluator kills at 150s)
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def objective_and_grad(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        Returns value and analytic gradient.
        """
        g = 1.0 - h
        c_all = np.concatenate([np.correlate(h, g, mode='full')[:N-1], 
                                 np.correlate(h, g, mode='full')[N:]])
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        lse = max_c + np.log(sum_exps) / beta
        
        # Gradient of LSE w.r.t. h
        # dLSE/dh_i = (1/sum_exps) * sum_k exps[k] * dC_k/dh_i
        # C_k = sum_j h[j] * (1 - h[j+k]) for positive shift k (linear, valid j range)
        # dC_k/dh_i = (1 - h[i+k]) if i+k in range, plus -h[i-k] if i-k in range, etc.
        # This is complex; use numerical gradient fallback but with better step
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        d_sum_pen = 2 * lambda_sum * (sum_h - N/2) * np.ones(N)
        
        val = lse + sum_pen
        
        # For gradient, we'll use a finite difference approach internally
        # but return None to let L-BFGS-B handle it (it's more robust)
        # Actually, let's compute analytic gradient properly
        
        # Recompute correlations efficiently
        g_vec = 1.0 - h
        # For each shift k (non-zero), C_k = sum_{valid j} h[j] * (1 - h[j+k])
        # We need dC_k/dh_i
        
        # Build gradient via correlation structure
        # For linear correlation, C_k for k>0: sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
        # dC_k/dh_i = (1-h[i+k]) if i <= N-1-k, plus -h[i-k] if i >= k and i-k <= N-1-k
        # This gets messy. Let's use a simpler approach:
        # Use the fact that the gradient of sum_k w_k * C_k where w_k = exps[k]/sum_exps / beta
        # is equivalent to a weighted correlation
        
        # Simpler: compute gradient numerically with central differences
        # but that's expensive. Let's just return None and trust L-BFGS-B
        return val, None

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
        Uses a more robust approach.
        """
        h = np.array(h, dtype=float).copy()
        
        # Iteratively adjust to satisfy sum constraint while respecting bounds
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N/2 - current_sum
            
            if abs(diff) < 1e-14:
                break
            
            # Determine which elements can be adjusted
            if diff > 0:
                # Need to increase sum: increase elements that are < 1
                mask = h < 1.0
                if not np.any(mask):
                    break
                n_incr = np.sum(mask)
                step = diff / n_incr
                h[mask] = np.minimum(h[mask] + step, 1.0)
            else:
                # Need to decrease sum: decrease elements that are > 0
                mask = h > 0.0
                if not np.any(mask):
                    break
                n_decr = np.sum(mask)
                step = diff / n_decr
                h[mask] = np.maximum(h[mask] + step, 0.0)
        
        # Final clip
        h = np.clip(h, 0.0, 1.0)
        
        # One more correction if needed
        current_sum = np.sum(h)
        if abs(current_sum - N/2) > 1e-10:
            # Distribute the remaining difference among elements not at bounds
            diff = N/2 - current_sum
            if diff > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        
        return h

    def optimize_instance(h0, beta, lambda_sum, max_iter, ftol=1e-8, gtol=1e-8):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            val, _ = objective_and_grad(h, beta, lambda_sum)
            return val

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)

    rng = np.random.default_rng(42)
    
    # List of restart configurations
    restarts = []
    
    # 1. Uniform start
    restarts.append((np.full(N, 0.5), 5.0, 100.0, 500))
    
    # 2-5. Random perturbations
    for i in range(4):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 5.0 + i * 5.0
        lambda_sum = 100.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))
        
    # 6-10. Bimodal initializations
    for i in range(5):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.9, 0.1)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 15.0 + i * 5.0
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))

    for idx, (h0, beta, lambda_sum, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        h_opt = optimize_instance(h0, beta, lambda_sum, max_iter)
        
        current_score = score_h(h_opt)
        
        if not np.isclose(np.sum(h_opt), N/2, atol=1e-6):
            continue
        if not (np.all(h_opt >= 0) and np.all(h_opt <= 1)):
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
    
    # Final polish pass with tighter tolerances
    if time.time() - start_time < time_limit - 10:
        # Use the best solution found so far as starting point
        h_polish = best_h.copy()
        
        # Try several high-beta polish iterations
        for beta_polish in [30.0, 50.0, 100.0]:
            if time.time() - start_time > time_limit:
                break
            
            h_opt = optimize_instance(h_polish, beta_polish, 500.0, 1000, ftol=1e-10, gtol=1e-10)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-8) and \
               np.all(h_opt >= 0) and np.all(h_opt <= 1):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    h_polish = h_opt.copy()
    
    return best_h
