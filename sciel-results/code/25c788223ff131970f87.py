"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time

N = 400
TIME_LIMIT = 145  # safety margin before 150s kill

def _compute_acf(h):
    """Compute the full autoconvolution (autocorrelation) via FFT."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _c3_bound(h):
    """Compute the C3 upper bound: max absolute value of autoconvolution."""
    acf = _compute_acf(h)
    # The autoconvolution values at lags 0 to N-1 (and symmetric)
    # We take max abs over all lags
    return np.max(np.abs(acf[:N]))

def _soft_max_psl(h, beta=10.0):
    """
    Smooth surrogate for max |acf| using log-sum-exp.
    This provides a differentiable approximation to the non-smooth max.
    """
    acf = _compute_acf(h)
    vals = np.abs(acf[:N])
    # Log-sum-exp for numerical stability
    m = np.max(vals)
    if m == 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (vals - m)))) / beta + m
    return log_sum

def _soft_max_grad(h, beta=10.0):
    """
    Gradient of the soft-max surrogate.
    d/dh_j [LSE(beta * |acf|)] = (1/beta) * sum_k exp(beta*|acf_k|) * d|acf_k|/dh_j
    """
    acf = _compute_acf(h)
    vals = np.abs(acf[:N])
    m = np.max(vals)
    if m == 0:
        return np.zeros(N)
    
    weights = np.exp(beta * (vals - m))
    weights /= np.sum(weights)
    
    # Gradient of |acf_k| with respect to h_j
    # acf_k = sum_i h_i * h_{i+k} (for the autoconvolution)
    # d|acf_k|/dh_j = sign(acf_k) * (h_{j+k} + h_{j-k}) if in range
    # More precisely: acf_k = sum_i h_i h_{i+k}
    # d acf_k / dh_j = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0 and j != j+k)
    # But since we're using the FFT-based acf, let's compute gradient via FFT
    
    # Actually, let's compute the gradient more carefully.
    # acf[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k] for k >= 0
    # d acf[k] / dh[j] = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
    # But this double-counts when j+k = j-k, i.e., k=0.
    # For k=0: acf[0] = sum h[i]^2, d acf[0]/dh[j] = 2*h[j]
    # For k>0: d acf[k]/dh[j] = h[j+k] + h[j-k] (with boundary conditions)
    
    # Let's compute the gradient of the soft-max directly.
    # grad_j = sum_k weights[k] * sign(acf[k]) * d acf[k] / dh[j]
    
    grad = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w < 1e-15:
            continue
        s = np.sign(acf[k]) if acf[k] != 0 else 0.0
        # d acf[k] / dh[j] for each j
        if k == 0:
            # d acf[0]/dh[j] = 2*h[j]
            grad += w * s * 2.0 * h
        else:
            # d acf[k]/dh[j] = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
            # For j in [0, N-1-k]: h[j+k]
            # For j in [k, N-1]: h[j-k]
            # So:
            #   for j in [0, N-1-k]: contribution h[j+k]
            #   for j in [k, N-1]: contribution h[j-k]
            #   for j in [N-k, N-1] (which is empty if k>=N, but k<N): only h[j-k] if j>=k
            # Let's just do it with shifts
            contrib = np.zeros(N)
            if N - k > 0:
                contrib[:N-k] += h[k:]
            if k < N:
                contrib[k:] += h[:N-k]
            grad += w * s * contrib
    
    return grad / beta  # The 1/beta factor from LSE derivative


def _gradient_step(h, lr=0.01, beta=10.0, steps=100):
    """
    Perform gradient descent on the continuous relaxation to minimize the soft-max surrogate.
    Then threshold to get signs.
    """
    x = h.astype(np.float64).copy()
    for step in range(steps):
        g = _soft_max_grad(x, beta=beta)
        x -= lr * g
        # Keep x in a reasonable range
        x = np.clip(x, -2.0, 2.0)
    # Threshold to get signs
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    return signs


def _local_search(h, max_iter=5000, time_limit=None):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    """
    h = np.array(h, dtype=np.float64)
    current_bound = _c3_bound(h)
    start = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize with random signs
    np.random.seed(42)
    best_h = np.ones(N)
    best_bound = _c3_bound(best_h)
    
    # Try a few random initializations
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Gradient-based refinement from best initialization
    if time.time() < deadline - 20:
        # Use gradient descent on continuous relaxation
        signs = _gradient_step(best_h, lr=0.05, beta=20.0, steps=200)
        bound = _c3_bound(signs)
        if bound < best_bound:
            best_bound = bound
            best_h = signs.copy()
        
        # Try different beta values
        for beta in [5.0, 15.0, 30.0]:
            if time.time() > deadline - 15:
                break
            signs = _gradient_step(best_h, lr=0.03, beta=beta, steps=100)
            bound = _c3_bound(signs)
            if bound < best_bound:
                best_bound = bound
                best_h = signs.copy()
    
    # Local search from best initialization
    if time.time() < deadline - 10:
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_limit=deadline - 5)
    
    # Additional restarts if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            # Gradient refine
            if time.time() < deadline - 10:
                signs = _gradient_step(h, lr=0.05, beta=20.0, steps=100)
                bound = _c3_bound(signs)
            else:
                signs = h
                bound = _c3_bound(h)
            
            if bound < best_bound:
                # Local search
                signs, bound = _local_search(signs, max_iter=1000, time_limit=deadline - 5)
                if bound < best_bound:
                    best_bound = bound
                    best_h = signs.copy()
    
    # Final polish: one more local search pass
    if time.time() < deadline - 3:
        best_h, best_bound = _local_search(best_h, max_iter=2000, time_limit=deadline - 1)
    
    # Final check
    final_bound = _c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_c3_bound(h)}")
    print(f"Score: {1.0 / _c3_bound(h)}")
