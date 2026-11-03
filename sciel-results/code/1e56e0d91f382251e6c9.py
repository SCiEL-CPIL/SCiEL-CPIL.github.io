import numpy as np
import time
import warnings

N = 50

def _compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)
    For non-negative h, sup|h*h| is the peak of the autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _soft_max_score_neg(h, temperature=1.0):
    """
    Negative soft-max score for minimization.
    Uses a soft-max surrogate for the autoconvolution max to make the objective smooth.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Soft max approximation
    if temperature <= 0:
        max_conv = np.max(conv)
    else:
        c_max = np.max(conv)
        # Log-sum-exp trick for numerical stability
        # max_conv approx = T * log(sum(exp(conv/T)))
        # = T * (c_max/T + log(sum(exp((conv-c_max)/T))))
        # = c_max + T * log(sum(exp((conv-c_max)/T)))
        log_sum_exp = np.log(np.sum(np.exp((conv - c_max) / temperature)))
        max_conv = c_max + temperature * log_sum_exp
        
    if max_conv < 1e-12:
        return 0.0
        
    score = s2 / (s1 * max_conv)
    return -score

def _soft_max_grad(h, temperature=1.0):
    """
    Compute gradient of the soft-max score w.r.t h.
    Score S = s2 / (s1 * M), where M = max(conv) approx.
    grad S = (2h * s1*M - s2 * (grad_s1 * M + s1 * grad_M)) / (s1*M)^2
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    
    if temperature <= 0:
        # Hard max gradient is tricky, use finite diff or just return zeros to be safe
        # But we can approximate by using the index of max
        idx_max = np.argmax(conv)
        weights = np.zeros_like(conv)
        weights[idx_max] = 1.0
    else:
        c_max = np.max(conv)
        # Weights for soft max: w_k = exp((conv_k - c_max)/T) / sum(exp(...))
        exp_terms = np.exp((conv - c_max) / temperature)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
    # grad s2 = 2h
    # grad s1 = 1
    # grad M = sum_k weights_k * grad conv_k
    # grad conv_k / grad h_i = h_{k-i} + h_{i-k} (if indices valid)
    
    grad_M = np.zeros(N)
    for k in range(len(conv)):
        if weights[k] < 1e-10:
            continue
        # conv[k] = sum_j h[j] h[k-j]
        # d/dh[i] conv[k] = h[k-i] + h[i-k]
        for i in range(N):
            idx1 = k - i
            if 0 <= idx1 < N:
                grad_M[i] += weights[k] * h[idx1]
            idx2 = i - k
            if 0 <= idx2 < N:
                grad_M[i] += weights[k] * h[idx2]
                
    # Simplify:
    # S = s2 / (s1 M)
    # dS/dh_i = [ (2 h_i) (s1 M) - s2 ( (1) M + s1 (grad_M_i) ) ] / (s1 M)^2
    
    denom = (s1 * (np.max(conv) if temperature <= 0 else np.max(conv) + temperature * np.log(np.sum(np.exp((conv - np.max(conv))/temperature)))))**2
    # Actually, let's just compute the scalar M value we used in the score
    if temperature <= 0:
        M_val = np.max(conv)
    else:
        c_max = np.max(conv)
        M_val = c_max + temperature * np.log(np.sum(np.exp((conv - c_max) / temperature)))
        
    denom = (s1 * M_val)**2
    
    term1 = 2.0 * h * (s1 * M_val)
    term2 = s2 * (M_val + s1 * grad_M)
    
    grad = (term1 - term2) / denom
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Strategy:
    # 1. Initialize with a Sidon-like set to get a good starting point (high L2, low peak relative to L2).
    #    A Sidon set (B_2 sequence) has distinct pairwise differences.
    #    For N=50, we can't fit a perfect Sidon set of size 50 in [0, 49] easily, but we can
    #    select a subset of indices that form a "sparse" pattern.
    #    However, the knowledge says: "Replacing the single spike with a maximal Sidon set of disjoint spikes
    #    increases the L2 norm relative to the autoconvolution peak under L1 normalization... Measured: 1 -> 1.173."
    #    This suggests that spreading mass over a Sidon set helps.
    
    # Let's construct a simple Sidon-like set.
    # A greedy Sidon set construction:
    # Start with empty set. Add smallest integer not creating duplicate differences.
    # But we have fixed 50 bins. We need to choose weights.
    # If we put mass only on Sidon indices, and 0 elsewhere, we have a sparse vector.
    # Let's try to find a subset of indices of size K (e.g., 10-15) that are Sidon-like.
    # Or, simply use a "sparse random" initialization and refine.
    
    # Better initialization:
    # Use a "Golomb ruler" like structure.
    # Let's pick indices: 0, 1, 3, 7, 15, 31... (powers of 2 minus 1?)
    # Or just use the first 10 indices? No, that creates many collisions.
    # Let's use a random sparse initialization and then optimize.
    
    np.random.seed(42)
    
    # Initialize with a few random spikes
    num_spikes = 5
    idxs = np.random.choice(N, num_spikes, replace=False)
    h_init = np.zeros(N)
    h_init[idxs] = 1.0
    h_init /= np.sum(h_init)
    
    r_init = _compute_ratio(h_init)
    if r_init > best_ratio:
        best_ratio = r_init
        best_h = h_init.copy()
        
    # Also check uniform
    h_uni = np.ones(N) / N
    r_uni = _compute_ratio(h_uni)
    if r_uni > best_ratio:
        best_ratio = r_uni
        best_h = h_uni.copy()
        
    # Also check single spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    r_spike = _compute_ratio(h_spike)
    if r_spike > best_ratio:
        best_ratio = r_spike
        best_h = h_spike.copy()

    # Optimization using Gradient Descent on the soft-max objective
    # We want to maximize Score = s2 / (s1 * M)
    # Subject to h >= 0.
    # We can work in log-space or use multiplicative updates.
    # Multiplicative update rule derived from gradient ascent on S:
    # h_i <- h_i * exp(eta * grad_S_i)
    # But we need to handle the s1 normalization.
    # If we fix s1 = 1, then S = s2 / M.
    # Let's fix s1 = 1 by normalizing after each step.
    
    current_h = best_h.copy()
    current_h /= np.sum(current_h)
    
    eta = 0.05
    max_iters = 2000
    patience = 100
    stale = 0
    last_best = best_ratio
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Compute gradient of the soft-max score
        temp = 0.1
        grad = _soft_max_grad(current_h, temp)
        
        # Multiplicative update
        # h_i <- h_i * (1 + eta * grad_i)  (First order approximation of exp)
        # Or h_i <- h_i * exp(eta * grad_i)
        
        update = np.exp(eta * grad)
        current_h = current_h * update
        
        # Clip to avoid negative values (shouldn't happen with exp, but just in case)
        current_h = np.maximum(current_h, 0)
        
        # Renormalize to sum=1
        s = np.sum(current_h)
        if s < 1e-15:
            break
        current_h /= s
        
        # Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
            if stale > patience:
                # Try a different initialization or restart
                # Random sparse restart
                num_spikes = np.random.randint(3, 15)
                idxs = np.random.choice(N, num_spikes, replace=False)
                current_h = np.zeros(N)
                current_h[idxs] = 1.0
                current_h /= np.sum(current_h)
                stale = 0
                # Also reduce eta
                eta *= 0.9
                if eta < 1e-4:
                    break
                    
    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
