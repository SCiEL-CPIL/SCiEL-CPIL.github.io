import numpy as np
import time

N = 100

def _compute_overlap_max(h):
    """Compute max overlap of h with shifted (1-h) for all cyclic shifts."""
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = 1 - autocorr(k)  (since sum h = N/2, and we normalize by N... wait)
    # Actually sum_i h[i] = N/2, so overlap(k) = N/2 - sum_i h[i]*h[(i+k)%N]
    # But we want the actual value, not normalized.
    # Let's compute directly using FFT for speed
    h_ext = np.concatenate([h, h])
    # For each shift k, overlap = sum_{i=0}^{N-1} h[i] * (1 - h[i+k])
    # = sum h[i] - sum h[i]*h[i+k]
    # sum h[i] = N/2
    # sum h[i]*h[i+k] is the circular autocorrelation
    # Use FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]  (circular)
    # overlap(k) = N/2 - autocorr[k]
    overlaps = N / 2.0 - autocorr
    return np.max(overslaps)

def _compute_overlap_max_v2(h):
    """Compute max overlap using direct FFT method."""
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    overlaps = N / 2.0 - autocorr
    return np.max(overlaps)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s
    
    # Start with a random initialization
    np.random.seed(42)
    h = np.random.uniform(0, 1, N)
    # Project to sum = N/2
    h = (h - h.mean()) * 0.5 + 0.5
    h = np.clip(h, 0, 1)
    # Renormalize
    s = h.sum()
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    best_h = h.copy()
    best_overlap = _compute_overlap_max_v2(best_h)
    
    # Gradient descent with soft-max smoothing
    # Objective: minimize max(overlaps) using log-sum-exp as smooth surrogate
    # f(h) = (1/beta) * log(sum(exp(beta * overlap_k)))
    
    beta = 10.0  # temperature for soft-max
    
    def soft_max_overlap(h):
        H = np.fft.fft(h)
        power = H * np.conj(H)
        autocorr = np.real(np.fft.ifft(power))
        overlaps = N / 2.0 - autocorr
        # log-sum-exp for numerical stability
        max_ov = np.max(overlaps)
        return max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    # Gradient via finite differences or analytic
    # Let's use a simple iterative approach with projection
    
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute current overlaps
        H = np.fft.fft(h)
        power = H * np.conj(H)
        autocorr = np.real(np.fft.ifft(power))
        overlaps = N / 2.0 - autocorr
        
        # Soft-max weight for each overlap
        max_ov = np.max(overlaps)
        weights = np.exp(beta * (overlaps - max_ov))
        weights /= np.sum(weights)
        
        # Gradient of overlap_k w.r.t. h:
        # overlap_k = N/2 - sum_i h[i]*h[(i+k)%N]
        # d(overlap_k)/dh[j] = -sum_i [delta_{i,j}*h[(i+k)%N] + h[i]*delta_{(i+k)%N,j}]
        # = -h[(j+k)%N] - h[(j-k)%N]
        # 
        # So gradient of soft-max objective:
        # grad_j = -sum_k weights[k] * (h[(j+k)%N] + h[(j-k)%N])
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] < 1e-10:
                continue
            wk = weights[k]
            # d(overlap_k)/dh[j] = -h[(j+k)%N] - h[(j-k)%N]
            # So contribution to grad[j] from weight wk: -wk * (-h[(j+k)%N] - h[(j-k)%N])
            # = wk * (h[(j+k)%N] + h[(j-k)%N])
            # Wait, we want to minimize, so gradient descent: h -= lr * grad
            # grad of soft-max = sum_k weights[k] * grad(overlap_k)
            # grad(overlap_k)[j] = -h[(j+k)%N] - h[(j-k)%N]
            # So total grad[j] = sum_k weights[k] * (-h[(j+k)%N] - h[(j-k)%N])
            # We want to minimize, so h_new = h - lr * grad = h + lr * sum_k weights[k]*(h[(j+k)%N]+h[(j-k)%N])
            
            shift_plus = np.roll(h, -k)  # h[(j+k)%N] for each j
            shift_minus = np.roll(h, k)   # h[(j-k)%N] for each j
            grad += wk * (-shift_plus - shift_minus)
        
        # Update
        h = h - lr * grad
        
        # Project to [0,1] and sum = N/2
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 1e-10:
            h = h * (N / 2.0) / s
        h = np.clip(h, 0, 1)
        
        # Check improvement
        cur_overlap = _compute_overlap_max_v2(h)
        if cur_overlap < best_overlap:
            best_overlap = cur_overlap
            best_h = h.copy()
        
        # Adaptive learning rate
        if iteration % 500 == 0 and iteration > 0:
            lr *= 0.9
    
    # Final polish: try a few random restarts if time permits
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline:
                break
            h = np.random.uniform(0, 1, N)
            h = (h - h.mean()) * 0.5 + 0.5
            h = np.clip(h, 0, 1)
            s = h.sum()
            if s > 0:
                h = h * (N / 2.0) / s
            h = np.clip(h, 0, 1)
            
            lr = 0.01
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                H = np.fft.fft(h)
                power = H * np.conj(H)
                autocorr = np.real(np.fft.ifft(power))
                overlaps = N / 2.0 - autocorr
                
                max_ov = np.max(overlaps)
                weights = np.exp(beta * (overlaps - max_ov))
                weights /= np.sum(weights)
                
                grad = np.zeros(N)
                for k in range(N):
                    if weights[k] < 1e-10:
                        continue
                    wk = weights[k]
                    shift_plus = np.roll(h, -k)
                    shift_minus = np.roll(h, k)
                    grad += wk * (-shift_plus - shift_minus)
                
                h = h - lr * grad
                h = np.clip(h, 0, 1)
                s = h.sum()
                if s > 1e-10:
                    h = h * (N / 2.0) / s
                h = np.clip(h, 0, 1)
                
                cur_overlap = _compute_overlap_max_v2(h)
                if cur_overlap < best_overlap:
                    best_overlap = cur_overlap
                    best_h = h.copy()
                
                if iteration % 400 == 0 and iteration > 0:
                    lr *= 0.9
    
    return best_h
