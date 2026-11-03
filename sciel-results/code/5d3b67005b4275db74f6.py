"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _logsumexp(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def _overlap_vector(h):
    """Compute overlap at all shifts: sum_i h_i * (1 - h_{(i+k) mod N})"""
    # 1-h
    one_minus_h = 1.0 - h
    # For each shift k, compute sum_i h_i * one_minus_h[(i+k) % N]
    # Use FFT for speed
    # correlation of h and one_minus_h
    # sum_i h_i * g[(i+k)%N] = circular cross-correlation
    # Use ifft(fft(h) * conj(fft(g)))
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    # cross-correlation: ifft(fft(h) * conj(fft(g))) gives sum_i h_i * conj(g[(i-k)%N])
    # We want sum_i h_i * g[(i+k)%N]
    # Let's compute: ifft(fft(h) * conj(fft(g)))_k = sum_i h_i * conj(g[(i-k)%N])
    # Since g is real, conj(g) = g
    # = sum_i h_i * g[(i-k)%N]
    # We want sum_i h_i * g[(i+k)%N] = sum_i h_i * g[(-i-k)%N] with index flip
    # Alternatively: ifft(fft(g) * conj(fft(h)))_k = sum_i g_i * h[(i-k)%N]
    # Let me just use direct computation or fix the FFT approach
    # Actually: circular cross-correlation r[k] = sum_i x[i] * y[(i+k) mod N]
    # = ifft(fft(x) * conj(fft(y))) is NOT right
    # Correct: r[k] = sum_i x[i] * y[(i-k) mod N] = ifft(fft(x) * conj(fft(y)))
    # So for sum_i h[i] * y[(i+k) mod N], let j = i+k, i = j-k
    # = sum_j h[j-k] * y[j] = ifft(fft(y) * conj(fft(h)))_k... 
    # Let me just do it directly for N=100, it's fast enough
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Start with constant 0.5
    h = np.full(N, 0.5)
    
    # Projected gradient descent with log-sum-exp smoothing
    # Objective: max_k overlap_k, smoothed via log-sum-exp
    # Minimize: logsumexp(overlap_k / tau) * tau, then project to [0,1] and sum=N/2
    
    best_h = h.copy()
    best_score = np.inf
    
    # Compute initial score
    ov = _overlap_vector(h)
    current_max = np.max(ov)
    best_score = current_max
    
    lr = 0.1
    tau = 0.1  # smoothing parameter, decrease over time
    
    # Use multiple phases with decreasing tau
    for phase in range(10):
        tau = max(0.01, 0.5 * (0.5 ** phase))
        n_iters = 500
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            ov = _overlap_vector(h)
            # Log-sum-exp gradient
            ov_max = np.max(ov)
            exp_ov = np.exp((ov - ov_max) / tau)
            weights = exp_ov / np.sum(exp_ov)
            
            # Gradient of logsumexp(ov/tau)*tau w.r.t. h
            # d/dh logsumexp(ov/tau) = sum_k weights_k * d(ov_k)/dh
            # ov_k = sum_i h_i * (1 - h[(i+k)%N])
            # d(ov_k)/dh_j = (1 - h[(j+k)%N]) - h_j * (1 if k==0 else 0)... 
            # Actually: ov_k = sum_i h_i - sum_i h_i * h[(i+k)%N] = N/2 - autocorr[k]
            # d(ov_k)/dh_j = 1 - sum_i [d(h_i * h[(i+k)%N])/dh_j]
            # = 1 - [h[(j+k)%N] + h[(j-k)%N]] (for the autocorrelation term)
            # Wait: autocorr[k] = sum_i h_i * h[(i+k)%N]
            # d(autocorr[k])/dh_j = h[(j+k)%N] + (if k!=0) h[(j-k)%N]
            # For k=0: d/dh_j sum_i h_i^2 = 2*h_j
            # So d(ov_k)/dh_j = 1 - h[(j+k)%N] - (1 if k!=0 else 1)*h[(j-k)%N]
            # For k=0: 1 - 2*h_j
            # For k!=0: 1 - h[(j+k)%N] - h[(j-k)%N]
            
            # Gradient of smoothed objective
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] < 1e-10:
                    continue
                if k == 0:
                    d_ov = 1.0 - 2.0 * h
                else:
                    d_ov = 1.0 - np.roll(h, -k) - np.roll(h, k)
                grad += weights[k] * d_ov
            
            grad *= tau  # chain rule for logsumexp(ov/tau)*tau
            
            # Projected gradient step
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            
            # Project to sum = N/2
            diff = N / 2.0 - np.sum(h_new)
            # Distribute correction
            h_new += diff / N
            h_new = np.clip(h_new, 0.0, 1.0)
            
            # Re-project to sum = N/2 (clip may have changed sum)
            for _ in range(10):
                diff = N / 2.0 - np.sum(h_new)
                if abs(diff) < 1e-12:
                    break
                h_new += diff / N
                h_new = np.clip(h_new, 0.0, 1.0)
            
            h = h_new
            
            # Check score
            ov = _overlap_vector(h)
            current_max = np.max(ov)
            if current_max < best_score:
                best_score = current_max
                best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Final projection to ensure sum = N/2 exactly
    best_h = np.clip(best_h, 0.0, 1.0)
    for _ in range(100):
        diff = N / 2.0 - np.sum(best_h)
        if abs(diff) < 1e-15:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    residual = N / 2.0 - np.sum(best_h)
    if abs(residual) > 1e-15:
        # Add residual to the element with most room
        idx = np.argmax(1.0 - best_h) if residual > 0 else np.argmin(best_h)
        best_h[idx] += residual
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-adjust
        diff = N / 2.0 - np.sum(best_h)
        if abs(diff) > 1e-15:
            best_h += diff / N
    
    return best_h
# EVOLVE-BLOCK-END
