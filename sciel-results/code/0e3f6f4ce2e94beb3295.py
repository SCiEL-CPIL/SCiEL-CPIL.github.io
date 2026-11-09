# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, SLSQP

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    X1, X2, X3, X4 are i.i.d. from the discrete measure defined by (support, weights).
    Uses sorted support for efficient pairwise sum CDF queries.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Ensure support is sorted for efficiency
    if not np.all(support[:-1] <= support[1:]):
        order = np.argsort(support)
        support = support[order]
        p = p[order]
    
    # Build all pairwise sums in a structured way
    # Since support is sorted, we can exploit the ordering
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort once
    sort_idx = np.argsort(pair_sums_flat, kind='stable')
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k, l) pair, we need CDF at threshold = 2*s_l - s_k
    # threshold[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # P = sum_{k,l} p_k * p_l * CDF(2*s_l - s_k)
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)

def _evaluate_from_params(params, n):
    """
    Evaluate measure from raw parameters: [s_0..s_{n-1}, w_0..w_{n-1}]
    """
    support = params[:n]
    weights = params[n:]
    
    # Clamp support to non-negative and sort
    support = np.maximum(support, 0.0)
    support = np.sort(support)
    
    # Clamp weights to non-negative and normalize
    weights = np.maximum(weights, 0.0)
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return 0.0
    p = weights / w_sum
    
    try:
        P = _evaluate_measure(support, p)
    except Exception:
        return 0.0
    
    return P

def _slsqp_objective(params, n):
    """
    Objective for SLSQP.
    params: [s_0, ..., s_{n-1}, log_w_0, ..., log_w_{n-1}]
    Returns -P (to minimize)
    """
    s = params[:n]
    log_w = params[n:]
    
    # Ensure support is non-negative
    s = np.maximum(s, 0.0)
    
    # Sort support? 
    # SLSQP assumes a continuous parameter space. Sorting introduces discontinuities.
    # However, if we start close to a sorted solution and constrain s_i <= s_{i+1}, 
    # the sorting is implicit in the constraints.
    # We must NOT sort inside the objective if we use constraints, 
    # otherwise the gradient is wrong when indices swap.
    # But if we don't sort, _evaluate_measure sorts it internally.
    # This breaks the gradient.
    
    # Strategy: 
    # 1. Optimize in a space where support is NOT sorted, but we add a penalty or constraint?
    # 2. Or, optimize gaps: s_0, d_1, ..., d_{n-1} where s_i = s_0 + sum(d_j).
    #    This ensures s is sorted if d_j >= 0.
    #    Let's use gaps.
    
    # Re-parameterize:
    # params: [s_0, d_1, ..., d_{n-1}, log_w_0, ..., log_w_{n-1}]
    # s_0 >= 0
    # d_i >= 0
    # log_w_i is free.
    
    s_0 = s[0]
    if n > 1:
        gaps = s[1:]
        s_sorted = np.empty(n)
        s_sorted[0] = s_0
        s_sorted[1:] = s_0 + np.cumsum(gaps)
    else:
        s_sorted = np.array([s_0])
        
    # Weights from log_w
    log_w_shifted = log_w - np.max(log_w)
    w = np.exp(log_w_shifted)
    w_sum = np.sum(w)
    if w_sum < 1e-9:
        return 0.0
    p = w / w_sum
    
    try:
        P = _evaluate_measure(s_sorted, p)
    except Exception:
        return 0.0
        
    return -P

def _slsqp_constraints(n):
    """
    Define constraints for SLSQP.
    1. s_0 >= 0
    2. d_i >= 0 for i=1..n-1
    """
    constraints = []
    # s_0 >= 0
    constraints.append({'type': 'ineq', 'fun': lambda x: x[0]})
    # d_i >= 0
    for i in range(1, n):
        idx = i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=idx: x[i]})
        
    return constraints

def _slsqp_bounds(n):
    """
    Bounds for SLSQP.
    s_0: [0, 10]
    d_i: [0, 10]
    log_w: [-10, 10]
    """
    bounds = [(0.0, 10.0)] * n + [(-10.0, 10.0)] * n
    return bounds

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 595  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    def time_left():
        return deadline - time.time()
    
    # Analytic 2-point optimum: support {0, 1}, weight p* on 1 where p* = (1+sqrt(17))/8
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # For differential evolution: minimize -P
    def objective(params, n):
        P = _evaluate_from_params(params, n)
        return -P
    
    # Try various sizes of support - focus more on smaller n for speed
    for n in [3, 4, 5, 6, 7]:
        if time_left() < 90:
            break
        
        # Bounds for support: [0, 5]
        # Bounds for weights: [0, 1]
        bounds = [(0.0, 5.0)] * n + [(0.0, 1.0)] * n
        
        # More restarts for smaller n (faster evaluations)
        n_restarts = 5 if n == 3 else (4 if n == 4 else (3 if n == 5 else 2))
        
        for restart in range(n_restarts):
            if time_left() < 45:
                break
            
            # Different initial populations
            if restart == 0:
                # Start near the 2-point solution, spread out
                init_s = np.sort(np.linspace(0.0, 1.0, n))
                init_w = np.ones(n) / n
            elif restart == 1:
                # Random initialization
                init_s = np.sort(np.random.uniform(0, 3, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n) * 2)
            elif restart == 2:
                # Another random initialization
                init_s = np.sort(np.random.uniform(0, 4, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n))
            elif restart == 3:
                # Concentrated initialization
                init_s = np.sort(np.concatenate([[0.0], np.random.uniform(0.5, 2.0, n-1)]))
                init_w = np.random.dirichlet(np.ones(n) * 5)
            else:
                # Wide spread
                init_s = np.sort(np.concatenate([[0.0], np.random.uniform(1.0, 5.0, n-1)]))
                init_w = np.random.dirichlet(np.ones(n) * 1.5)
            
            x0 = np.concatenate([init_s, init_w])
            
            # Run differential evolution with adjusted parameters
            maxiter = 200 if n <= 4 else 150
            popsize = 25 if n <= 4 else 20
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    args=(n,),
                    x0=x0,
                    seed=42 + restart * 100 + n * 1000,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    polish=False,
                    init='sobol' if restart == 0 else 'latinhypercube',
                    mutation=(0.5, 1.0),
                    recombination=0.9
                )
            except Exception:
                continue
            
            x_opt = result.x
            P_opt = _evaluate_from_params(x_opt, n)
            
            support_opt = np.maximum(x_opt[:n], 0.0)
            support_opt = np.sort(support_opt)
            w_opt = np.maximum(x_opt[n:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            update_best(support_opt, p_opt, P_opt)
    
    # Try to grow the solution: add a new point and optimize
    if best_support is not None and time_left() > 60:
        n_best = len(best_support)
        n_grow = n_best + 1
        
        # Insert a new point between existing ones or at the end
        candidates = []
        if n_best >= 2:
            mid = (best_support[-1] - best_support[0]) / 2
            candidates.append(mid)
        candidates.append(best_support[-1] + 1.0)
        candidates.append(0.1)
        
        for new_point in candidates:
            if time_left() < 30:
                break
            
            # Build initial support by inserting new point
            init_s = np.sort(np.concatenate([best_support, [new_point]]))
            # Start with small weight on new point
            init_w = np.concatenate([best_weights, [0.05]])
            w_sum_init = np.sum(init_w)
            init_w = init_w / w_sum_init
            
            x0 = np.concatenate([init_s, init_w])
            bounds = [(0.0, 5.0)] * n_grow + [(0.0, 1.0)] * n_grow
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    args=(n_grow,),
                    x0=x0,
                    seed=42 + n_grow * 10000,
                    maxiter=150,
                    popsize=20,
                    tol=1e-10,
                    polish=False,
                    init='latinhypercube',
                    mutation=(0.5, 1.0),
                    recombination=0.9
                )
            except Exception:
                continue
            
            x_opt = result.x
            P_opt = _evaluate_from_params(x_opt, n_grow)
            
            support_opt = np.maximum(x_opt[:n_grow], 0.0)
            support_opt = np.sort(support_opt)
            w_opt = np.maximum(x_opt[n_grow:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            update_best(support_opt, p_opt, P_opt)
    
    # Final local polish with SLSQP on the best solution
    # We use a gap-based parameterization to ensure sorting and non-negativity
    if best_support is not None and time_left() > 20:
        n_best = len(best_support)
        
        # Convert best_support to gaps
        # s_0, d_1, ..., d_{n-1}
        s_best = best_support
        gaps = np.diff(s_best)
        # Ensure gaps are non-negative (they are, since sorted)
        # Add a small epsilon to avoid zero gaps if they are exactly 0?
        # SLSQP can handle 0, but log-weights need care.
        
        # Convert best_weights to log-weights
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        
        # Initial params for SLSQP: [s_0, d_1, ..., d_{n-1}, log_w_0, ..., log_w_{n-1}]
        x0_slsqp = np.concatenate([[s_best[0]], gaps, log_w_best])
        
        # Define objective and constraints for SLSQP
        def slsqp_obj(params):
            return _slsqp_objective(params, n_best)
            
        cons = _slsqp_constraints(n_best)
        bnds = _slsqp_bounds(n_best)
        
        # Run SLSQP
        # Use multiple passes with tighter tolerances
        for polish_pass in range(3):
            if time_left() < 10:
                break
            
            # Increase iterations and tighten tolerance
            maxiter = 1000 if polish_pass == 0 else 2000
            ftol = 1e-12 if polish_pass == 0 else (1e-14 if polish_pass == 1 else 1e-16)
            
            try:
                result_slsqp = minimize(
                    slsqp_obj,
                    x0_slsqp,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons,
                    options={'maxiter': maxiter, 'ftol': ftol}
                )
            except Exception:
                result_slsqp = None
            
            if result_slsqp is not None and result_slsqp.success:
                x_opt_slsqp = result_slsqp.x
                s_0 = x_opt_slsqp[0]
                gaps = x_opt_slsqp[1:n_best]
                log_w = x_opt_slsqp[n_best:]
                
                # Reconstruct support
                support_final = np.empty(n_best)
                support_final[0] = s_0
                if n_best > 1:
                    support_final[1:] = s_0 + np.cumsum(gaps)
                
                # Reconstruct weights
                log_w_shifted = log_w - np.max(log_w)
                w_final = np.exp(log_w_shifted)
                w_sum_final = np.sum(w_final)
                if w_sum_final >= 1e-9:
                    p_final = w_final / w_sum_final
                    score_final = _evaluate_measure(support_final, p_final)
                    update_best(support_final, p_final, score_final)
                    
                    # Update x0_slsqp for next pass
                    x0_slsqp = x_opt_slsqp

    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity: no NaN, no inf, non-negative, positive sum
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=0.0, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=0.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    w_sum = np.sum(best_weights)
    if w_sum < 1e-9:
        best_support = support_two
        best_weights = weights_two
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
