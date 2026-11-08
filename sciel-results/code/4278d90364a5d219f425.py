"""Combined program: FFT-based overlap computation with multi-start L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_max_overlap_fft(h):
    """
    Compute max_k sum_i h[i] * (1 - h[(i+k) % N]) using FFT.
    This is the periodic overlap metric.
    """
    B = 1.0 - h
    # Circular cross-correlation: overlap[k] = sum_i h[i] * B[(i+k) % N]
    # Using FFT: overlap = ifft(fft(h) * conj(fft(B))) gives sum_i h[i] * conj(B[i-k])
    # But for real signals, conj(B) = B, so we need to be careful with indexing.
    # Actually, let's use the direct formula via FFT:
    # overlap[k] = sum_i h[i] * B[i+k mod N]
    # This is the circular cross-correlation of h and B.
    # In numpy, np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(B))) gives:
    # c[k] = sum_i h[i] * conj(B[i-k mod N]) = sum_i h[i] * B[i-k mod N] (for real B)
    # We want overlap[k] = sum_i h[i] * B[i+k mod N]
    # So overlap[k] = c[-k mod N] = c[N-k mod N]
    H = np.fft.fft(h)
    Bf = np.fft.fft(B)
    corr = np.fft.ifft(H * np.conj(Bf)).real
    # corr[k] = sum_i h[i] * B[i-k mod N]
    # We want overlap[k] = sum_i h[i] * B[i+k mod N] = corr[-k mod N] = corr[N-k]
    # So overlap = corr flipped: overlap[k] = corr[N-k] for k=0..N-1
    # corr[0] = sum h[i]*B[i], corr[1] = sum h[i]*B[i-1], etc.
    # overlap[0] = sum h[i]*B[i] = corr[0]
    # overlap[1] = sum h[i]*B[i+1] = corr[N-1]
    # overlap[k] = corr[N-k] for k>=1, corr[0] for k=0
    overlap = np.zeros(N)
    overlap[0] = corr[0]
    overlap[1:] = corr[:0:-1]  # corr[N-1], corr[N-2], ..., corr[1]
    return np.max(overlap)

def _compute_max_overlap(h):
    """
    Compute max_k sum_i h[i] * (1 - h[(i+k) % N]).
    Uses FFT for speed.
    """
    return _compute_max_overlap_fft(h)

def _soft_max_overlap(h, beta=10.0):
    """
    Smooth surrogate for max overlap using log-sum-exp.
    overlap[k] = sum_i h[i] * (1-h[(i+k)%N])
    soft_max = (1/beta) * log(sum(exp(beta * overlap[k])))
    """
    B = 1.0 - h
    H = np.fft.fft(h)
    Bf = np.fft.fft(B)
    corr = np.fft.ifft(H * np.conj(Bf)).real
    overlap = np.zeros(N)
    overlap[0] = corr[0]
    overlap[1:] = corr[:0:-1]
    # Soft max via log-sum-exp
    max_ov = np.max(overlap)
    if max_ov > 100:  # Avoid overflow
        return max_ov + np.log(np.sum(np.exp(beta * (overlap - max_ov)))) / beta
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * overlap)))

def _objective_smooth(x, beta=10.0):
    """Smooth objective: soft max overlap + penalty for sum deviation"""
    h = np.clip(x, 0, 1)
    sum_dev = np.sum(h) - N / 2
    penalty = 50.0 * sum_dev ** 2
    ov = _soft_max_overlap(h, beta=beta)
    return ov + penalty

def _objective_grad_smooth(x, beta=10.0):
    """Numerical gradient of _objective_smooth"""
    eps = 1e-5
    grad = np.zeros(N)
    f0 = _objective_smooth(x, beta)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective_smooth(x_plus, beta)
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = _objective_smooth(x_minus, beta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def _adjust_sum(h, target_sum=N/2):
    """Adjust h to have sum exactly target_sum while keeping values in [0,1]"""
    h = h.copy()
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-10:
        return h
    # Distribute diff across elements that can be adjusted
    if diff > 0:
        # Need to increase some values
        indices = np.where(h < 1.0)[0]
        if len(indices) == 0:
            return h
        # Sort by how much room they have
        room = 1.0 - h[indices]
        order = np.argsort(-room)
        remaining = diff
        for idx in indices[order]:
            adj = min(remaining, room[idx] - 1e-10)
            if adj > 0:
                h[idx] += adj
                remaining -= adj
            if remaining < 1e-10:
                break
    else:
        # Need to decrease some values
        indices = np.where(h > 0.0)[0]
        if len(indices) == 0:
            return h
        # Sort by how much they can be decreased
        room = h[indices]
        order = np.argsort(-room)
        remaining = -diff
        for idx in indices[order]:
            adj = min(remaining, room[idx] - 1e-10)
            if adj > 0:
                h[idx] -= adj
                remaining -= adj
            if remaining < 1e-10:
                break
    return h

def construct():
    start_time = time.time()
    
    # Start with uniform 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _compute_max_overlap(h0)
    
    # Multi-start L-BFGS-B with smooth objective
    np.random.seed(42)
    
    max_iterations = 300
    
    for restart in range(15):
        if time.time() > DEADLINE - 10:
            break
        
        # Random initialization
        if restart == 0:
            h_init = h0.copy()
        else:
            # Try different perturbation scales
            scale = 0.1 + 0.05 * (restart % 4)
            h_init = np.full(N, 0.5) + np.random.randn(N) * scale
            h_init = np.clip(h_init, 0, 1)
            h_init = _adjust_sum(h_init)
        
        # Run L-BFGS-B with smooth objective
        result = minimize(
            _objective_smooth,
            h_init,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': max_iterations, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        h_opt = np.clip(result.x, 0, 1)
        h_opt = _adjust_sum(h_opt)
        
        val = _compute_max_overlap(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
        
        # Check time
        if time.time() > DEADLINE - 15:
            break
    
    # Final polish: local search with exact overlap evaluation
    if time.time() < DEADLINE - 5:
        current_h = best_h.copy()
        current_val = best_val
        
        # Try multiple local search strategies
        for search_iter in range(500):
            if time.time() > DEADLINE - 2:
                break
            
            # Random pair swap to maintain sum
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            delta = np.random.uniform(-0.05, 0.05)
            
            new_h = current_h.copy()
            new_h[idx1] = np.clip(new_h[idx1] + delta, 0, 1)
            new_h[idx2] = np.clip(new_h[idx2] - delta, 0, 1)
            
            # Check if sum is preserved
            if abs(np.sum(new_h) - N/2) > 1e-9:
                continue
            
            s = _compute_max_overlap(new_h)
            if s < current_val - 1e-10:
                current_h = new_h
                current_val = s
                if s < best_val:
                    best_val = s
                    best_h = new_h.copy()
    
    # Ensure final result has sum exactly N/2
    final_h = best_h.copy()
    final_h = _adjust_sum(final_h)
    
    return final_h
