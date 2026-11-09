# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums s_i + s_j with probabilities p_i * p_j
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    # Cumulative sum of probs
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds shape (n, n): thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # Find index where sorted_sums < T_flat (strict inequality)
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # p_k p_l
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
    
    def time_left():
        return deadline - time.time()

    # Strategy: Differential Evolution over (support, log_weights)
    # We parameterize weights using log-space to ensure positivity and sum-to-1 via softmax.
    # Support is constrained to [0, max_s].
    
    # We will try several sizes n.
    # For each n, we define the parameter vector x of length 2n:
    # x[0:n] = support points (constrained to [0, max_s])
    # x[n:2n] = log_weights (unconstrained, mapped to weights via softmax)
    
    # To make the search space more uniform, we can scale the log_weights.
    # Let raw_w = x[n:2n]. weights = exp(raw_w) / sum(exp(raw_w)).
    # This is equivalent to softmax.
    
    # Objective function for DE: minimizes -P
    def objective(x, n):
        if time_left() < 5:
            return 0.0 # Return low score if time is up, DE will stop or we catch it
        
        support = x[:n]
        log_w = x[n:]
        
        # Softmax for weights
        # Subtract max for numerical stability
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        weights = exp_w / np.sum(exp_w)
        
        # Ensure support is sorted? 
        # The objective _evaluate_measure doesn't strictly require sorted support, 
        # but it's good for consistency. However, permuting support and weights 
        # together yields the same P. DE might find unsorted supports.
        # _evaluate_measure handles any order.
        
        # Ensure support >= 0. DE bounds handle this.
        
        try:
            p_val = _evaluate_measure(support, weights)
        except:
            return 0.0
            
        return -p_val

    # Define bounds for DE
    # Support: [0, max_s]
    # Log-weights: [-log_w_range, log_w_range]
    # A range of [-10, 10] for log-weights allows ratios of e^20 ~ 5e8, which is plenty.
    
    max_s = 10.0
    log_w_bound = 10.0
    
    # Try different n values
    # n=2 is trivial and known. n=3, 4, 5 are promising.
    # DE is expensive, so we limit population size and generations.
    
    n_values = [3, 4, 5, 6]
    
    # Budget allocation:
    # Each DE run should not take too long.
    # Let's aim for ~100-200 evaluations per n value if possible, or adapt.
    # DE with popsize=10, maxiter=20 might be ~200 evals.
    
    for n in n_values:
        if time_left() < 30:
            break
            
        bounds = []
        # Support bounds
        for i in range(n):
            bounds.append((0.0, max_s))
        # Log weight bounds
        for i in range(n):
            bounds.append((-log_w_bound, log_w_bound))
        
        # DE parameters
        # popsize: factor * n_params. n_params = 2n.
        # For n=3, n_params=6. popsize=10 -> 60 individuals.
        # maxiter: 20 -> 1200 evals. Might be too slow if eval is slow.
        # Let's monitor time.
        
        popsize = 10
        maxiter = 20
        tol = 1e-4
        
        # Initial best for this n
        best_local_score = -1.0
        best_local_x = None
        
        # Use a callback to check time and update best
        def callback(xk, convergence):
            nonlocal best_local_score, best_local_x
            if time_left() < 5:
                # Stop DE by raising exception? Or just return False?
                # differential_evolution doesn't support early stopping via callback return value directly in all versions, 
                # but we can check time in objective.
                # Actually, returning False from callback doesn't stop it.
                # We rely on objective returning 0 and time checks.
                pass
            
            # Evaluate current best xk? No, xk is current population member? 
            # No, xk is the current best in the population at this iteration.
            # Let's evaluate it to update global best.
            support = xk[:n]
            log_w = xk[n:]
            max_log_w = np.max(log_w)
            exp_w = np.exp(log_w - max_log_w)
            weights = exp_w / np.sum(exp_w)
            
            try:
                p_val = _evaluate_measure(support, weights)
            except:
                p_val = 0.0
                
            if p_val > best_local_score:
                best_local_score = p_val
                best_local_x = xk.copy()
                
            # Update global best
            if p_val > best_score:
                update_best(support, weights, p_val)
                
            return False # Do not stop

        # Run DE
        # To avoid long hangs, we wrap it.
        # DE can be slow. We'll set a relatively small maxiter and popsize.
        
        # If n is large, DE is slower.
        # Let's adjust maxiter based on n.
        if n <= 4:
            maxiter = 30
        else:
            maxiter = 20
            
        try:
            result = differential_evolution(
                objective,
                bounds,
                args=(n,),
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=42,
                polish=False, # Polish can be slow and gradient-based, DE is robust enough
                init='latinhypercube'
            )
        except Exception as e:
            print(f"DE failed for n={n}: {e}")
            continue
            
        # After DE, evaluate the final result
        if result.x is not None:
            support = result.x[:n]
            log_w = result.x[n:]
            max_log_w = np.max(log_w)
            exp_w = np.exp(log_w - max_log_w)
            weights = exp_w / np.sum(exp_w)
            
            try:
                p_val = _evaluate_measure(support, weights)
            except:
                p_val = 0.0
                
            if p_val > best_score:
                update_best(support, weights, p_val)

    # Fallback if nothing found or if DE failed to improve
    if best_support is None or best_score < 0.5:
        # Use a known good baseline: 2-point distribution
        # P(p) for support [0,1] and weights [1-p, p]
        # Max at p approx 0.38? 
        # Let's just use the 2-point optimal found by simple search
        for p in np.linspace(0.0, 1.0, 101):
            support = np.array([0.0, 1.0])
            weights = np.array([1.0 - p, p])
            score = _evaluate_measure(support, weights)
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = weights.copy()
                
        if best_support is None:
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])
            best_score = _evaluate_measure(best_support, best_weights)

    return best_support, best_weights

# EVOLVE-BLOCK-END
