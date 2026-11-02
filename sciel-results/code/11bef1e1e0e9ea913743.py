import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    return conv

def _score_and_grad(heights, beta=100.0):
    """Compute smooth surrogate score and gradient.
    
    Uses log-sum-exp to approximate the max in the autoconvolution,
    making the objective differentiable.
    """
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    total = np.sum(heights)
    if total <= 0:
        return np.inf, np.zeros(n)
    
    # Normalize conv by total^2 * w / n to get the normalized quantity
    # score = max_conv * n / (total^2 * w)
    # So normalized conv = conv * n / (total^2 * w)
    norm_conv = conv * n / (total * total * w)
    
    # Use log-sum-exp to approximate max
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    # For stability, shift by max
    max_val = np.max(norm_conv)
    shifted = beta * (norm_conv - max_val)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_val + log_sum_exp / beta
    
    # Gradient of smooth_max w.r.t. heights
    # d/dh [max_val + (1/beta) log(sum(exp(beta*(norm_conv - max_val))))]
    # = (1/beta) * sum(exp(beta*(norm_conv_i - max_val))) / sum(exp(beta*(norm_conv_i - max_val))) * d(norm_conv_i)/dh
    # = sum(exp(shifted_i)) / sum(exp(shifted)) * d(norm_conv_i)/dh
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # d(norm_conv)/dh = d(conv * n / (total^2 * w)) / dh
    # = (n / (total^2 * w)) * d(conv)/dh - (conv * n * 2 * total / (total^4 * w)) * d(total)/dh
    # = (n / (total^2 * w)) * d(conv)/dh - (2 * conv * n / (total^3 * w)) * d(total)/dh
    
    # d(conv)/dh: conv is autoconvolution, so d(conv_k)/dh_j = h_{k-j} + h_{j-k} (with appropriate bounds)
    # But we can compute this more efficiently using FFT derivatives
    # Actually, let's just compute the gradient numerically or use a simpler approach
    
    # For now, use finite differences for gradient (slower but reliable)
    # Or we can use the fact that grad = weights @ (d norm_conv / dh)
    
    # Let's compute d(conv)/dh analytically
    # conv = h * h (linear convolution)
    # d(conv)/dh = 2 * h (as a linear operator, it's the convolution with h, times 2)
    # So d(conv_k)/dh_j = h_{k-j} + h_{j-k} where indices are valid
    
    # This is equivalent to: d(conv)/dh = 2 * (h convolved with h reversed? No)
    # Actually, conv_k = sum_j h_j * h_{k-j}
    # d(conv_k)/dh_j = h_{k-j} + h_{j-k} (by product rule, since h appears twice)
    # But h_{j-k} = h_{-(k-j)} which is only valid if we extend h to be zero outside [0,n)
    
    # Let's just use the FFT to compute the gradient of conv w.r.t. h
    # grad_conv = 2 * ifft(fft(h) * fft(h_reversed))? No, that's not right.
    
    # Actually, the gradient of sum_k conv_k * alpha_k w.r.t. h_j is:
    # sum_k alpha_k * (h_{k-j} + h_{j-k}) = (alpha * h)_j + (alpha * h_flipped)_j
    # where * denotes cross-correlation
    
    # So d(conv)/dh as a linear operator applied to alpha is:
    # alpha * h + alpha * flip(h)
    
    # Let's compute this
    # weights are in the conv domain (length M_FFT, but only first 2n-1 are meaningful)
    # We need to compute: grad_h = (n/(total^2*w)) * (weights_conv * h + weights_conv * flip(h))
    #                    - (2*n/(total^3*w)) * (conv * weights_conv) * (1/total) * ... 
    
    # This is getting complicated. Let's just use finite differences for the gradient
    # with a small step, which is O(n) FFTs but we can batch them
    
    # Actually, for a cleaner approach, let's just return the smooth score and use
    # a finite-difference gradient in the optimizer
    
    return smooth_max, None

def _score_and_grad_fd(heights, beta=100.0, eps=1e-6):
    """Compute smooth surrogate score and finite-difference gradient."""
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    total = np.sum(heights)
    if total <= 0:
        return np.inf, np.zeros(n)
    
    norm_conv = conv * n / (total * total * w)
    
    max_val = np.max(norm_conv)
    shifted = beta * (norm_conv - max_val)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_val + log_sum_exp / beta
    
    # Finite difference gradient
    grad = np.zeros(n)
    for j in range(n):
        h_plus = heights.copy()
        h_plus[j] += eps
        f_p = np.zeros(M_FFT, dtype=np.float64)
        f_p[:n] = h_plus
        F_p = np.fft.rfft(f_p)
        conv_p = np.fft.irfft(F_p * F_p, n=M_FFT)
        total_p = np.sum(h_plus)
        norm_conv_p = conv_p * n / (total_p * total_p * w)
        max_val_p = np.max(norm_conv_p)
        shifted_p = beta * (norm_conv_p - max_val_p)
        log_sum_exp_p = np.log(np.sum(np.exp(shifted_p)))
        smooth_max_p = max_val_p + log_sum_exp_p / beta
        grad[j] = (smooth_max_p - smooth_max) / eps
    
    return smooth_max, grad

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered
    x = np.linspace(0, 1, N)
    for width in [0.1, 0.2, 0.3, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks
    for left_frac in [0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.1, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Multi-Start Simulated Annealing directly on 600D heights
    num_chains = 5
    steps_per_chain = 2000
    
    for chain_idx in range(num_chains):
        if deadline_reached():
            break
        
        if chain_idx == 0:
            current_h = best_h.copy()
        else:
            if rng.random() < 0.5:
                current_h = candidates[rng.integers(0, len(candidates))].copy()
            else:
                current_h = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
            
        current_score = compute_score(current_h)
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        T_start = 0.1
        T_end = 0.001
        step_size = 0.1
        
        for step in range(steps_per_chain):
            if deadline_reached():
                break
                
            progress = step / steps_per_chain
            T = T_start * (T_end / T_start) ** progress
            
            noise = rng.normal(0, step_size, N)
            new_h = current_h + noise
            new_h = np.maximum(new_h, 0.0)
            
            if np.sum(new_h) <= 0:
                continue
                
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                accept = True
            else:
                delta = new_score - current_score
                if delta < 0.05: 
                    accept = rng.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = max(0.01, step_size * 0.999)
            else:
                step_size = max(0.01, step_size * 0.95)
                
    # Final polish: Local random search around best_h
    if time_left() > 10:
        polish_steps = 500
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.05
        
        for _ in range(polish_steps):
            if deadline_reached():
                break
            noise = rng.normal(0, step_size, N)
            new_h = np.maximum(current_h + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.1, step_size * 1.05)
            else:
                step_size = max(0.005, step_size * 0.9)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
