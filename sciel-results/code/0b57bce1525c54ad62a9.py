import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.special import softmax

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case, though optimization should keep them normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Compute all pairwise sums for X1 + X2
    # s_i + s_j
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    # Flatten and sort
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # 2. Compute P = sum_{k,l} w_k w_l F(2*s_l - s_k)
    # Thresholds matrix T[k, l] = 2*s_l - s_k
    # Note: s is (n,).
    # T shape (n, n) where T[k, l] corresponds to X3=s_k, X4=s_l
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find cumulative weight of pair_sums < threshold
    # searchsorted returns index of first element >= x.
    # We want < x, so index is count of elements < x.
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative weights
    # If idx > 0, cum_weight is cum_weights[idx-1]
    # If idx == 0, cum_weight is 0
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    # w[:, None] is (n, 1), w[None, :] is (1, n). Product is (n, n)
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def _compute_P_smooth_and_grad(params, n, tau):
    """
    Compute smoothed P and gradient w.r.t params.
    params: [log_w (n), log_s_diffs (n)]
    log_w: log weights, softmax gives w
    log_s_diffs: log differences, s[0] = exp(log_s[0]), s[i] = s[i-1] + exp(log_s[i])
    """
    log_w = params[:n]
    log_s = params[n:]
    
    w = softmax(log_w)
    
    # Reconstruct support
    s = np.empty(n)
    s[0] = np.exp(log_s[0])
    for i in range(1, n):
        s[i] = s[i-1] + np.exp(log_s[i])
        
    # Compute smoothed P
    # P = sum_{i,j,k,l} w_i w_j w_k w_l sigma( (2 s_l - s_i - s_j - s_k) / tau )
    
    # To be efficient, we can use matrix operations.
    # Let A = s_i + s_j + s_k. We need sigma((2 s_l - A)/tau).
    # This is an n^4 sum. For n=20, 160k terms. Feasible.
    # For n=50, 6.25M terms. Feasible in numpy.
    
    # Construct all sums s_i + s_j + s_k
    # s_i: (n, 1, 1, 1)
    # s_j: (1, n, 1, 1)
    # s_k: (1, 1, n, 1)
    # s_l: (1, 1, 1, n)
    
    si = s[:, None, None, None]
    sj = s[None, :, None, None]
    sk = s[None, None, :, None]
    sl = s[None, None, None, :]
    
    wi = w[:, None, None, None]
    wj = w[None, :, None, None]
    wk = w[None, None, :, None]
    wl = w[None, None, None, :]
    
    diffs = (2.0 * sl - si - sj - sk) / tau
    sigs = 1.0 / (1.0 + np.exp(-diffs))
    
    P_smooth = np.sum(wi * wj * wk * wl * sigs)
    
    # Gradient w.r.t params
    # dP/dlog_w_m = sum_{i,j,k,l} (delta_{m,i} w_j w_k w_l + ... ) sigs * w_i w_j w_k w_l ?
    # Actually, dP/dw_m = sum_{i,j,k,l} (delta_{m,i} w_j w_k w_l + delta_{m,j} w_i w_k w_l + delta_{m,k} w_i w_j w_l + delta_{m,l} w_i w_j w_k) sigs
    # Then dP/dlog_w_m = w_m * (dP/dw_m - sum_m' dP/dw_m' w_m')? No, softmax gradient.
    # dP/dlog_w_m = w_m * (dP/dw_m - sum_l dP/dw_l w_l) is for normalized?
    # Let's just compute dP/dw and then chain rule.
    # dP/dw_m = sum_{i,j,k,l} (delta_{m,i} w_j w_k w_l + delta_{m,j} w_i w_k w_l + delta_{m,k} w_i w_j w_l + delta_{m,l} w_i w_j w_k) sigs
    
    # Term for w_i (index m):
    # Sum over j,k,l: w_j w_k w_l sigs[m, j, k, l]
    # This is (sigs[m, :, :, :] * wj * wk * wl).sum()
    
    grad_w = np.zeros(n)
    
    # To save memory/time, we can compute these sums efficiently.
    # sigs is (n, n, n, n).
    # grad_w[m] = sum_{j,k,l} w_j w_k w_l sigs[m, j, k, l]  (from w_i)
    #             + sum_{i,k,l} w_i w_k w_l sigs[i, m, k, l]  (from w_j)
    #             + sum_{i,j,l} w_i w_j w_l sigs[i, j, m, l]  (from w_k)
    #             + sum_{i,j,k} w_i w_j w_k sigs[i, j, k, m]  (from w_l)
    
    # Due to symmetry of w and structure, these 4 terms are related.
    # Term 1: T1[m] = sum_{j,k,l} w_j w_k w_l sigs[m, j, k, l]
    # Term 2: T2[m] = sum_{i,k,l} w_i w_k w_l sigs[i, m, k, l]
    # Term 3: T3[m] = sum_{i,j,l} w_i w_j w_l sigs[i, j, m, l]
    # Term 4: T4[m] = sum_{i,j,k} w_i w_j w_k sigs[i, j, k, m]
    
    # We can compute T1 by contracting axes 1,2,3 of sigs with w.
    # T1 = np.einsum('ijkl,jkl->i', sigs, wj, wk, wl) -- wait, wj is (1,n,1,1)
    # T1 = np.tensordot(sigs, w, axes=([1,2,3],[0,0,0]))? No.
    # T1[m] = sum_{j,k,l} sigs[m,j,k,l] * w[j] * w[k] * w[l]
    # This is a tensor contraction.
    
    # Let's use einsum for clarity and speed.
    # sigs: (i, j, k, l)
    # w: (n,)
    
    T1 = np.einsum('ijkl,j,k,l->i', sigs, w, w, w, optimize=True)
    T2 = np.einsum('ijkl,i,k,l->j', sigs, w, w, w, optimize=True)
    T3 = np.einsum('ijkl,i,j,l->k', sigs, w, w, w, optimize=True)
    T4 = np.einsum('ijkl,i,j,k->l', sigs, w, w, w, optimize=True)
    
    grad_w = T1 + T2 + T3 + T4
    
    # Gradient w.r.t log_w
    # dP/dlog_w_m = w_m * (grad_w[m] - sum_k grad_w[k] w[k])
    # Note: sum_k grad_w[k] w[k] = dP/d(sum w) = 0? No, P is homogeneous degree 2? No, degree 4.
    # P(cw) = c^4 P(w).
    # Euler's theorem: sum_m w_m dP/dw_m = 4 P.
    # So sum_k grad_w[k] w[k] = 4 P_smooth.
    # Thus dP/dlog_w_m = w_m * (grad_w[m] - 4 P_smooth)
    
    grad_log_w = w * (grad_w - 4.0 * P_smooth)
    
    # Gradient w.r.t s
    # P = sum w_i w_j w_k w_l sigma(diffs)
    # dP/ds_m = sum (w_i w_j w_k w_l) * sigma'(diffs) * d(diffs)/ds_m
    # sigma'(x) = sigma(x)(1-sigma(x))
    # d(diffs)/ds_m:
    # diffs = (2 s_l - s_i - s_j - s_k) / tau
    # d/ds_m diffs = (2 delta_{m,l} - delta_{m,i} - delta_{m,j} - delta_{m,k}) / tau
    
    sig_prime = sigs * (1.0 - sigs)
    
    # dP/ds_m = sum_{i,j,k,l} w_i w_j w_k w_l sig_prime[i,j,k,l] * (2 delta_{m,l} - delta_{m,i} - delta_{m,j} - delta_{m,k}) / tau
    
    # Term for delta_{m,l}: 2 * sum_{i,j,k} w_i w_j w_k sig_prime[i,j,k,m]
    # Term for delta_{m,i}: - sum_{j,k,l} w_j w_k w_l sig_prime[m,j,k,l]
    # Term for delta_{m,j}: - sum_{i,k,l} w_i w_k w_l sig_prime[i,m,k,l]
    # Term for delta_{m,k}: - sum_{i,j,l} w_i w_j w_l sig_prime[i,j,m,l]
    
    # These are similar to T1..T4 but with sig_prime.
    S1 = np.einsum('ijkl,j,k,l->i', sig_prime, w, w, w, optimize=True)
    S2 = np.einsum('ijkl,i,k,l->j', sig_prime, w, w, w, optimize=True)
    S3 = np.einsum('ijkl,i,j,l->k', sig_prime, w, w, w, optimize=True)
    S4 = np.einsum('ijkl,i,j,k->l', sig_prime, w, w, w, optimize=True)
    
    grad_s = (2.0 * S4 - S1 - S2 - S3) / tau
    
    # Gradient w.r.t log_s
    # s[0] = exp(log_s[0])
    # s[i] = s[i-1] + exp(log_s[i])
    # d s[m] / d log_s[m] = exp(log_s[m]) for m=0?
    # s[0] = e^a0. ds0/da0 = e^a0 = s0.
    # s[i] = s[i-1] + e^ai. ds[i]/da[i] = e^ai = s[i] - s[i-1].
    # But s[m] depends on a[m] and a[m-1]...
    # Actually, s[k] = sum_{j=0}^k exp(a_j).
    # d s[k] / d a_m = exp(a_m) if m <= k, else 0.
    # So dP/da_m = sum_{k=m}^{n-1} grad_s[k] * exp(a_m)
    #             = exp(a_m) * sum_{k=m}^{n-1} grad_s[k]
    
    grad_log_s = np.zeros(n)
    suffix_sum = 0.0
    for m in range(n-1, -1, -1):
        suffix_sum += grad_s[m]
        grad_log_s[m] = np.exp(log_s[m]) * suffix_sum
        
    grad_params = np.concatenate([grad_log_w, grad_log_s])
    
    return -P_smooth, grad_params

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    def update_best(s, w, score):
        nonlocal best_support, best_weights, best_score
        if score > best_score:
            best_support = s.copy()
            best_weights = w.copy()
            best_score = score

    # Strategy: Smoothed joint optimization of support and weights
    # Use L-BFGS-B on log-parameters
    
    n_values = [5, 8, 12, 16, 20]
    taus = [0.1, 0.5, 1.0, 5.0]
    
    for n in n_values:
        if time.time() > deadline - 10.0:
            break
            
        n_restarts = 10 if n <= 12 else 5
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize log_w and log_s
            # Random Dirichlet for w
            w_init = np.random.dirichlet(np.ones(n))
            log_w_init = np.log(w_init + 1e-12)
            
            # Random positive support
            # Log-spaced or random
            if r % 2 == 0:
                s_init = np.linspace(0.1, 5.0, n)
            else:
                s_init = np.sort(np.random.uniform(0.1, 5.0, n))
            
            # Convert to log_s diffs
            log_s_init = np.zeros(n)
            log_s_init[0] = np.log(s_init[0])
            for i in range(1, n):
                log_s_init[i] = np.log(s_init[i] - s_init[i-1] + 1e-6)
            
            x0 = np.concatenate([log_w_init, log_s_init])
            
            # Optimize with multiple taus
            for tau in taus:
                if time.time() > deadline - 10.0:
                    break
                
                try:
                    # Use L-BFGS-B
                    result = minimize(_compute_P_smooth_and_grad, x0, args=(n, tau), 
                                      method='L-BFGS-B', 
                                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
                    
                    if result.success or result.fun < _compute_P_smooth_and_grad(x0, n, tau)[0]:
                        x_opt = result.x
                        log_w_opt = x_opt[:n]
                        log_s_opt = x_opt[n:]
                        
                        w_opt = softmax(log_w_opt)
                        s_opt = np.empty(n)
                        s_opt[0] = np.exp(log_s_opt[0])
                        for i in range(1, n):
                            s_opt[i] = s_opt[i-1] + np.exp(log_s_opt[i])
                        
                        # Check validity
                        if np.any(s_opt < 0) or np.any(w_opt < 0):
                            continue
                        
                        score = _compute_P(s_opt, w_opt)
                        update_best(s_opt, w_opt, score)
                        
                except Exception:
                    pass

    # Fallback if no valid solution found
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
