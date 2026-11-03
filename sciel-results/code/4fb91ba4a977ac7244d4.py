import numpy as np
import time
import random

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
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

def _conv_max_at_lag(h, lag):
    """Compute sum_i h[i]*h[i+lag] for a given lag (can be negative)."""
    n = len(h)
    if lag >= 0:
        if lag >= n:
            return 0.0
        return np.sum(h[:n-lag] * h[lag:])
    else:
        l = -lag
        if l >= n:
            return 0.0
        return np.sum(h[l:] * h[:n-l])

def _softmax_weights(conv, temperature):
    """Compute softmax weights over the convolution values."""
    # conv has length 2N-1, indices 0 to 2N-2
    # We want to weight the lags that are close to the max
    max_conv = np.max(conv)
    # Subtract max for numerical stability
    shifted = conv - max_conv
    # Scale by temperature
    weights = np.exp(shifted / temperature)
    total = np.sum(weights)
    if total < 1e-15:
        # Fallback to uniform
        weights = np.ones(len(conv)) / len(conv)
    else:
        weights /= total
    return weights

def _gradient_of_conv_at_lag(h, lag):
    """
    Compute gradient of conv[lag] w.r.t h.
    conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] (with h=0 outside [0,N-1])
    d conv[k] / d h[i] = h[k-i] + h[i-k] (if indices valid)
    """
    n = len(h)
    grad = np.zeros(n)
    for i in range(n):
        val = 0.0
        idx1 = lag - i
        if 0 <= idx1 < n:
            val += h[idx1]
        idx2 = i - lag
        if 0 <= idx2 < n:
            val += h[idx2]
        grad[i] = val
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    def update_best(h):
        nonlocal best_h, best_ratio
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
        return r

    # Initialize with uniform distribution
    h = np.ones(N) / N
    update_best(h)
    
    # Also check delta as baseline
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    update_best(h_delta)

    # Strategy: Iterative optimization with smooth surrogate
    # We use a stochastic subgradient method with softmax-weighted lags
    # to approximate the gradient of the supremum norm.
    
    eta = 0.05  # Learning rate
    temperature = 0.1  # Temperature for softmax
    max_iters = 8000
    patience = 100
    stale = 0
    
    current_h = h.copy()
    current_ratio = _compute_ratio(current_h)
    
    # Use a random seed for reproducibility in the search
    rng = np.random.RandomState(42)
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # 1. Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv < 1e-15:
            break
        
        # 2. Compute softmax weights over the convolution
        weights = _softmax_weights(conv, temperature)
        
        # 3. Compute weighted gradient
        # grad_h[i] = sum_k weights[k] * d conv[k] / d h[i]
        grad_h = np.zeros(N)
        for k in range(len(conv)):
            if weights[k] > 1e-10:
                grad_k = _gradient_of_conv_at_lag(current_h, k - (N-1))
                # Wait, the lag indexing: conv[k] corresponds to lag = k - (N-1)
                # Let's verify: for N=50, conv has length 99.
                # conv[0] = h[0]*h[0], lag = 0? No.
                # np.convolve(h, h, 'full')[k] = sum_j h[j] h[k-j]
                # For k=0: h[0]h[0], this is lag 0.
                # For k=1: h[0]h[1] + h[1]h[0], this is lag 1? 
                # Actually, the standard autocorrelation lag m is sum h[i] h[i+m].
                # np.convolve(h, h, 'full') gives the convolution, which is the same as correlation for real signals but with a flip.
                # Let's stick to the gradient formula derived earlier.
                # d conv[k] / d h[i] = h[k-i] + h[i-k]
                # This is correct for the convolution definition.
                grad_h += weights[k] * grad_k
        
        # 4. Multiplicative Update
        # We want to minimize sup_conv relative to ||h||_2^2.
        # The gradient points in the direction of increasing conv.
        # So we subtract eta * grad_h.
        # To maintain non-negativity and scale, we use a multiplicative update.
        # h_i <- h_i * exp(-eta * grad_h_i)
        
        # To prevent underflow/overflow, we normalize the gradient first
        max_grad = np.max(np.abs(grad_h))
        if max_grad > 1e-15:
            grad_h = grad_h / max_grad
        
        update = np.exp(-eta * grad_h)
        current_h = current_h * update
        
        # Renormalize to sum=1
        s = np.sum(current_h)
        if s < 1e-15:
            # Restart
            current_h = np.ones(N) / N
            stale = 0
            continue
        current_h /= s
        
        # 5. Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
        
        # Adaptive temperature: decrease to focus on the peak
        if i % 50 == 0 and i > 0:
            temperature *= 0.95
            if temperature < 0.01:
                temperature = 0.01
        
        # If we stagnate, try a different strategy
        if stale > patience:
            # Random restart with a different pattern
            if i % 200 == 0:
                # Sparse random initialization
                mask = rng.rand(N) > 0.8
                if not np.any(mask):
                    mask[0] = True
                current_h = mask.astype(float)
                current_h /= np.sum(current_h)
                stale = 0
            else:
                # Decay eta
                eta *= 0.95
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
