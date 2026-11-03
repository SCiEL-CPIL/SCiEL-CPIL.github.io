import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Computes the score: L2^2 / (L1 * Sup(AutoConv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    
    if autoconv_max == 0:
        return 0.0
        
    return l2_sq / (l1 * autoconv_max)

def _active_set_convolution(h, top_k=5):
    """
    Identify the dominant convolution lags that constrain the peak.
    Returns indices of the top_k convolution peaks.
    """
    h_conv = np.convolve(h, h, mode='full')
    # Get indices of top_k peaks
    top_indices = np.argsort(h_conv)[-top_k:]
    return top_indices

def _objective_trust_region(h, tau):
    """
    Objective for trust-region optimization.
    We minimize -Score = -(L2^2) / (L1 * DELTA * MaxConv)
    
    Using smooth approximation for max convolution.
    """
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    
    # Soft max approximation with numerical stability
    max_val = np.max(h_conv)
    if max_val < 1e-15:
        return 0.0
    
    # Log-sum-exp for smooth max
    shifted = (h_conv - max_val) / tau
    log_sum_exp = max_val + tau * np.log(np.sum(np.exp(shifted)))
    smooth_max_conv = log_sum_exp
    
    if smooth_max_conv < 1e-10:
        return 0.0
        
    score = l2_sq / (l1 * DELTA * smooth_max_conv)
    return -score

def _gradient_trust_region(h, tau):
    """
    Compute gradient of the smooth objective.
    """
    l1 = np.sum(h)
    if l1 < 1e-10:
        return np.zeros(N)
    
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    
    max_val = np.max(h_conv)
    if max_val < 1e-15:
        return np.zeros(N)
    
    shifted = (h_conv - max_val) / tau
    exp_shifted = np.exp(shifted)
    log_sum_exp = max_val + tau * np.log(np.sum(exp_shifted))
    smooth_max_conv = log_sum_exp
    
    if smooth_max_conv < 1e-10:
        return np.zeros(N)
    
    score = l2_sq / (l1 * DELTA * smooth_max_conv)
    
    # Gradient of -score
    # d/dh [-L2^2 / (L1 * DELTA * MaxConv)]
    # = -(2*h*L1*MaxConv - L2^2*Delta*MaxConv_grad*L1 - L2^2*L1_grad*DELTA*MaxConv) / (L1^2 * DELTA^2 * MaxConv^2)
    
    # L1_grad = ones
    l1_grad = np.ones(N)
    
    # MaxConv_grad: derivative of log-sum-exp
    # d/dh_i [log sum exp((conv_j - max)/tau)] = (1/tau) * sum_j exp((conv_j - max)/tau) * dconv_j/dh_i
    # conv_j = sum_k h_k * h_{j-k}
    # dconv_j/dh_i = h_{j-i} + h_{i} (if j-i and i are valid indices)
    
    max_conv_grad = np.zeros(N)
    for i in range(N):
        # For each h_i, compute sum over all j of exp_shifted[j] * dconv_j/dh_i
        # dconv_j/dh_i = h_{j-i} (if 0 <= j-i < N) + h_i (if 0 <= i < N, which is always true)
        # Actually conv is full convolution, so we need to be careful
        for j in range(len(h_conv)):
            # conv_j = sum_k h_k * h_{j-k} where h is extended with zeros
            # dconv_j/dh_i = h_{j-i} + h_{i} (but only if j-i is in [0,N-1] and i is in [0,N-1])
            # More precisely: dconv_j/dh_i = h_{j-i} if 0 <= j-i < N else 0
            # Plus: if we consider h_i appears as h_k where k=i, then h_{j-i}
            # And if we consider h_{j-k} where j-k=i, then k=j-i, so h_k = h_{j-i}
            # Wait, let me reconsider.
            # conv_j = sum_k h_k * h_{j-k}
            # dconv_j/dh_i = h_{j-i} (from the first term when k=i) + h_{j-i} (from the second term when j-k=i, i.e., k=j-i)
            # But this double counts if j-i = i, i.e., j=2i
            # Actually no: d/dh_i [sum_k h_k * h_{j-k}] = h_{j-i} + sum_k h_k * d/dh_i h_{j-k}
            # = h_{j-i} + h_{j-i} (when k such that j-k=i, i.e., k=j-i)
            # = 2*h_{j-i} if j-i is in [0,N-1]
            # Wait, that's not right either. Let me be more careful.
            # conv_j = sum_k h_k * h_{j-k}
            # The derivative with respect to h_i:
            # For the term h_k where k=i: derivative is h_{j-i}
            # For the term h_{j-k} where j-k=i, i.e., k=j-i: derivative is h_k = h_{j-i}
            # So total = 2*h_{j-i} if 0 <= j-i < N
            # But if j-i = i (i.e., j=2i), then both terms are the same, so it's still 2*h_i
            # Actually no, when j=2i, k=i gives h_i * h_i, and d/dh_i = 2*h_i
            # When j != 2i, k=i gives h_i * h_{j-i}, d/dh_i = h_{j-i}
            # And k=j-i gives h_{j-i} * h_i, d/dh_i = h_{j-i}
            # So total = 2*h_{j-i} if j-i != i, and 2*h_i if j-i = i
            # In both cases: 2*h_{j-i} if 0 <= j-i < N
            
            if 0 <= j - i < N:
                max_conv_grad[i] += exp_shifted[j] * 2 * h[j - i]
    
    max_conv_grad /= tau
    
    # Now compute gradient of -score
    # score = l2_sq / (l1 * DELTA * smooth_max_conv)
    # -score = -l2_sq / (l1 * DELTA * smooth_max_conv)
    # d(-score)/dh_i = -(2*h_i * l1 * smooth_max_conv - l2_sq * l1_grad_i * DELTA * smooth_max_conv - l2_sq * l1 * DELTA * max_conv_grad_i) / (l1^2 * DELTA^2 * smooth_max_conv^2)
    # = -(2*h_i * l1 * smooth_max_conv - l2_sq * DELTA * smooth_max_conv - l2_sq * l1 * DELTA * max_conv_grad_i) / (l1^2 * DELTA^2 * smooth_max_conv^2)
    
    numerator = 2 * h * l1 * smooth_max_conv - l2_sq * DELTA * smooth_max_conv - l2_sq * l1 * DELTA * max_conv_grad
    denominator = l1**2 * DELTA**2 * smooth_max_conv**2
    
    grad = -numerator / denominator
    
    return grad

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussian peaks
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N/2, 15):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = np.sum(h_g)
            if s > 0:
                inits.append(h_g / s)
                
    # 3. Triangular peaks
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(2, N, 15):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            s = np.sum(h_t)
            if s > 0:
                inits.append(h_t / s)
                
    # 4. Sparse "Sidon-like" sets
    def greedy_sidon(n, k):
        S = []
        used_sums = set()
        for i in range(n):
            if len(S) >= k:
                break
            potential_sums = set()
            for s in S:
                potential_sums.add(i + s)
            potential_sums.add(i + i) # 2*i
            
            if not any(ps in used_sums for ps in potential_sums):
                S.append(i)
                used_sums.update(potential_sums)
        return S

    for k in range(3, 16):
        if time.time() > deadline: break
        try:
            S = greedy_sidon(N, k)
            if len(S) < 2: continue
            h_s = np.zeros(N)
            h_s[S] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
        except Exception:
            pass

    # 5. Random Sparse with varying densities
    np.random.seed(42)
    for _ in range(100):
        if time.time() > deadline: break
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_r = np.zeros(N)
        h_r[idx] = np.random.rand(k)
        s = np.sum(h_r)
        if s > 0:
            inits.append(h_r / s)

    # 6. Alternating patterns
    for period in [2, 3, 4, 5, 6, 7]:
        h_a = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h_a[i] = 1.0
        s = np.sum(h_a)
        if s > 0:
            inits.append(h_a / s)
            
    # 7. "Two peaks" patterns
    for c1 in np.linspace(2, N/2-2, 10):
        for c2 in np.linspace(N/2+2, N-2, 10):
            h_p = np.zeros(N)
            h_p[int(c1)] = 1.0
            h_p[int(c2)] = 1.0
            s = np.sum(h_p)
            if s > 0:
                inits.append(h_p / s)

    # Evaluate all initializations
    for h_init in inits:
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()

    # Optimization using trust-region approach with active-set identification
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Sort initializations by score
    init_scores = [(_score_raw(h), h) for h in inits]
    init_scores.sort(key=lambda x: -x[0])
    
    top_inits = [h for s, h in init_scores[:20]]
    if len(top_inits) == 0:
        top_inits = [best_h / np.sum(best_h)]
    else:
        if not any(np.allclose(h, best_h) for h in top_inits):
            top_inits.append(best_h / np.sum(best_h))
            
    # Trust-region optimization with annealing tau
    for h_init in top_inits:
        if time.time() > deadline - 15: break
        
        h_init = h_init / np.sum(h_init)
        
        try:
            from scipy.optimize import minimize
            
            # Annealing tau from larger to smaller
            taus = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
            current_h = h_init.copy()
            
            for tau in taus:
                if time.time() > deadline - 10: break
                
                # Use L-BFGS-B with trust-region-like behavior via tight bounds
                # First, identify active set (dominant convolution lags)
                active_lags = _active_set_convolution(current_h, top_k=5)
                
                # Create a modified objective that emphasizes the active lags
                def obj_active(h, tau=tau, active_lags=active_lags):
                    l1 = np.sum(h)
                    if l1 < 1e-10:
                        return 0.0
                    l2_sq = np.sum(h**2)
                    h_conv = np.convolve(h, h, mode='full')
                    
                    # Weighted max: emphasize active lags
                    weights = np.ones(len(h_conv))
                    weights[active_lags] = 10.0  # Emphasize dominant lags
                    
                    # Weighted log-sum-exp
                    weighted_conv = h_conv * weights
                    max_val = np.max(weighted_conv)
                    if max_val < 1e-15:
                        return 0.0
                    
                    shifted = (weighted_conv - max_val) / tau
                    log_sum_exp = max_val + tau * np.log(np.sum(np.exp(shifted)))
                    smooth_max_conv = log_sum_exp / 10.0  # Normalize back
                    
                    if smooth_max_conv < 1e-10:
                        return 0.0
                        
                    score = l2_sq / (l1 * DELTA * smooth_max_conv)
                    return -score
                
                res = minimize(obj_active, current_h, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success:
                    h_candidate = res.x
                    h_candidate = np.clip(h_candidate, 0, None)
                    s_norm = np.sum(h_candidate)
                    if s_norm > 0:
                        h_candidate /= s_norm
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
                    current_h = h_candidate
        except Exception:
            pass

    # Final Polish: use SLSQP with very tight tau for precision
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            from scipy.optimize import minimize
            res = minimize(lambda h: _objective_trust_region(h, 1e-4), h_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-15})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s_norm = np.sum(h_candidate)
                if s_norm > 0:
                    h_candidate /= s_norm
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Block-coordinate descent on symmetric pairs around dominant peak
    if time.time() < deadline - 3:
        h_current = best_h.copy()
        h_current /= np.sum(h_current)
        
        # Identify the dominant convolution lag
        h_conv = np.convolve(h_current, h_current, mode='full')
        dominant_lag = np.argmax(h_conv)
        
        # Try block-coordinate updates on pairs symmetric around the dominant lag
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for i in range(N // 2):
                j = N - 1 - i
                if i >= j:
                    continue
                
                # Try adjusting h[i] and h[j] while keeping sum constant
                h_ij_sum = h_current[i] + h_current[j]
                
                # Try a few different splits
                for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                    h_test = h_current.copy()
                    h_test[i] = h_ij_sum * alpha
                    h_test[j] = h_ij_sum * (1 - alpha)
                    
                    s_test = _score_raw(h_test)
                    if s_test > best_score + 1e-10:
                        best_score = s_test
                        best_h = h_test.copy()
                        h_current = h_test
                        improved = True
                        break
                    elif s_test > _score_raw(h_current):
                        # Accept if it improves current, even if not global best
                        h_current = h_test
            
            if not improved:
                break

    return list(best_h)
