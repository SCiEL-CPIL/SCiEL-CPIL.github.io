"""Optimized program for autocorrelation inequality task."""
import numpy as np
import time

N = 400

def _value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconv_and_grad(h):
    """Compute autoconvolution and subgradient of max(autoconv) w.r.t. h."""
    conv = np.convolve(h, h)
    k = np.argmax(conv)
    # conv[k] = sum_i h[i]*h[k-i] (with appropriate bounds)
    # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N else 0
    # So gradient is h reversed and appropriately shifted
    # conv[k] = sum_{i} h[i] * h[k-i] where h is defined on [0,N-1]
    # d/dh[j] conv[k] = h[k-j] if 0 <= k-j < N
    # So grad[j] = h[k-j] for j such that 0<=k-j<N, i.e., j in [max(0,k-N+1), min(N-1,k)]
    grad = np.zeros(N)
    lo = max(0, k - N + 1)
    hi = min(N - 1, k)
    for j in range(lo, hi + 1):
        grad[j] = h[k - j]
    return conv, k, grad

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Start with uniform (score 2.0), normalize sum to N
    h = np.ones(N)
    best_h = h.copy()
    best_val = _value(h)
    
    # Subgradient descent on max(autoconv) with sum(h)=N constraint
    # We minimize 2*max(autoconv)/N subject to sum(h)=N
    # Scale-invariant, so fix sum(h)=N
    
    lr = 1.0 / (np.sqrt(N) + 1)  # Initial learning rate
    momentum = 0.9
    vel = np.zeros(N)
    
    for it in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Ensure sum constraint
        h = h - np.mean(h) + 1.0  # project to sum=N
        
        conv, k, grad = _autoconv_and_grad(h)
        current_max = conv[k]
        current_val = 2.0 * current_max / N
        
        # Momentum update
        vel = momentum * vel + (1 - momentum) * grad
        h = h - lr * vel
        
        # Project back to sum(h) = N
        h = h - np.mean(h) + 1.0
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Decay learning rate
        lr = 1.0 / (np.sqrt(it + N) + 1)
        
        # Check convergence
        if it > 1000 and abs(current_val - best_val * N / 2.0) < 1e-10:
            break
    
    # Final polish: try to further reduce with smaller steps
    if time.time() < deadline - 10:
        lr2 = lr * 0.1
        for it in range(20000):
            if time.time() > deadline - 5:
                break
            h = best_h - np.mean(best_h) + 1.0
            conv, k, grad = _autoconv_and_grad(h)
            vel = 0.9 * vel + 0.1 * grad
            h = h - lr2 * vel
            h = h - np.mean(h) + 1.0
            val = _value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            lr2 *= 0.999
    
    return best_h
