import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 600
DEADLINE = 110.0  # Safety margin before 120s kill

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
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    n = len(heights)
    grad = np.zeros(n, dtype=np.float64)
    
    i_min = max(0, peak_idx - n + 1)
    i_max = min(n - 1, peak_idx)
    if i_min <= i_max:
        indices = np.arange(i_min, i_max + 1)
        j_indices = peak_idx - indices
        grad[indices] = 2.0 * heights[j_indices]
    
    return conv, grad, peak_idx

def _compute_softmax_score_and_grad(h, alpha, M):
    """Compute soft-max score and its gradient w.r.t. h.
    
    Score = max_conv * N / (S^2 * w) where S = sum(h), w = 0.5/N
    We use soft-max to smooth the max operation.
    """
    f = np.zeros(M, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    # Soft-max of conv
    cmax = np.max(conv)
    if cmax < 0:
        cmax = 0.0
    exp_conv = np.exp(alpha * (conv - cmax))
    soft_max = cmax + np.log(np.sum(exp_conv)) / alpha
    weights = exp_conv / np.sum(exp_conv)
    
    # Gradient of soft_max w.r.t. f: 
    # d(soft_max)/d(f_i) = sum_k weights_k * d(conv_k)/d(f_i) = 2 * (weights * f)[i] via correlation
    # conv_k = sum_j f_j * f_{k-j}, so d(conv_k)/d(f_i) = f_{k-i} + f_{k-i} = 2*f_{k-i} (when valid)
    # Sum_k weights_k * 2*f_{k-i} = 2 * sum_k weights_k * f_{k-i}
    # This is 2 * (weights * f)_i (circular conv of weights with f, evaluated at i)
    W = np.fft.rfft(weights)
    corr = np.fft.irfft(W * np.conj(F), n=M)
    grad_softmax_f = 2.0 * corr[:N]
    
    S = np.sum(h)
    w = 0.5 / N
    if S <= 1e-10:
        return 1e10, np.zeros(N)
    
    # score = soft_max * N / (S^2 * w)
    # d(score)/d(h_i) = N/w * [d(soft_max)/d(h_i) * 1/S^2 + soft_max * d(1/S^2)/d(h_i)]
    # d(1/S^2)/d(h_i) = -2/S^3
    # So: d(score)/d(h_i) = N/w * [grad_softmax_f[i]/S^2 - 2*soft_max/S^3]
    #                         = N/w/S^2 * [grad_softmax_f[i] - 2*soft_max/S]
    
    score = soft_max * N / (S * S * w)
    grad = (N / (w * S * S)) * (grad_softmax_f - 2.0 * soft_max * h / S)
    
    return score, grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # Phase 1: Structured candidates
    if time_left() > 30:
        x = np.linspace(0, 1, N)
        structured_candidates = []
        
        # Tapered edges (cosine taper)
        for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            h_taper = np.ones(N)
            taper_len = int(taper_frac * N)
            if taper_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
                h_taper[:taper_len] = ramp
                h_taper[-taper_len:] = ramp[::-1]
            structured_candidates.append(h_taper)
        
        # Gaussian bump
        for width in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            h_gauss = np.exp(-((x - 0.5) / width) ** 2)
            structured_candidates.append(h_gauss)
        
        # Parabolic
        for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_par = 1.0 - a * (2 * x - 1) ** 2
            h_par = np.maximum(h_par, 0.0)
            if np.max(h_par) > 0:
                structured_candidates.append(h_par)
        
        # Linear taper
        for slope in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_lin = 1.0 - slope * np.abs(2 * x - 1)
            h_lin = np.maximum(h_lin, 0.0)
            if np.max(h_lin) > 0:
                structured_candidates.append(h_lin)
        
        # Plateau with smooth edges
        for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h_plat = np.ones(N)
            edge_len = int((1 - plateau_frac) * N / 2)
            if edge_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
                h_plat[:edge_len] = ramp
                h_plat[-edge_len:] = ramp[::-1]
            structured_candidates.append(h_plat)
        
        # Two-block: separated rectangles
        for b1 in [50, 100, 150, 200]:
            for b2 in [50, 100, 150, 200]:
                if b1 + b2 > N:
                    continue
                for ratio in [0.5, 1.0, 2.0]:
                    h_two = np.zeros(N)
                    h_two[:b1] = 1.0
                    h_two[N - b2:] = ratio
                    structured_candidates.append(h_two)
        
        for h_cand in structured_candidates:
            if time_left() < 5:
                break
            h_cand = np.maximum(h_cand, 0.0)
            if np.sum(h_cand) <= 0:
                continue
            score = compute_score(h_cand)
            if score < best_score:
                best_score = score
                best_h = h_cand.copy()
    
    # Phase 2: Gradient-based optimization with soft-max surrogate
    if time_left() > 25:
        h_opt = best_h.copy()
        alpha = 30.0
        lr = 0.005
        patience = 0
        
        for it in range(500):
            if time_left() < 10:
                break
            
            obj, grad = _compute_softmax_score_and_grad(h_opt, alpha, M)
            
            # Normalize gradient by its norm for stable steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            # Adaptive learning rate
            h_new = h_opt - lr * grad
            h_new = np.maximum(h_new, 0.0)
            
            new_score = compute_score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h_opt = h_new.copy()
                patience = 0
            else:
                patience += 1
                # If stuck, try smaller step or restart from best
                if patience > 20:
                    h_opt = best_h.copy()
                    lr = lr * 0.7
                    patience = 0
            
            if it % 25 == 0:
                alpha = min(alpha * 1.3, 300.0)
                lr = max(lr * 0.95, 0.0005)
    
    # Phase 3: Random restarts with gradient optimization
    if time_left() > 20:
        rng = np.random.default_rng(42)
        
        for restart in range(8):
            if time_left() < 12:
                break
            
            # Random smooth initialization
            freqs = np.arange(N)
            h_init = np.ones(N)
            num_modes = rng.integers(2, 6)
            for _ in range(num_modes):
                k = rng.integers(1, N // 6)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.1, 0.4)
                h_init += amp * np.cos(2 * np.pi * k * freqs / N + phase)
            h_init = np.maximum(h_init, 0.0)
            if np.sum(h_init) <= 0:
                continue
            
            h_r = h_init.copy()
            alpha_r = 30.0
            lr_r = 0.005
            
            for it in range(150):
                if time_left() < 12:
                    break
                
                obj_r, grad_r = _compute_softmax_score_and_grad(h_r, alpha_r, M)
                
                h_r_new = h_r - lr_r * grad_r
                h_r_new = np.maximum(h_r_new, 0.0)
                
                score_r = compute_score(h_r_new)
                if score_r < best_score:
                    best_score = score_r
                    best_h = h_r_new.copy()
                    h_r = h_r_new.copy()
                
                if it % 20 == 0:
                    alpha_r = min(alpha_r * 1.2, 200.0)
                    lr_r = max(lr_r * 0.93, 0.001)
    
    # Phase 4: Final high-precision polish
    if time_left() > 8:
        h_final = best_h.copy()
        alpha_f = 150.0
        lr_f = 0.001
        
        for it in range(100):
            if time_left() < 5:
                break
            
            obj_f, grad_f = _compute_softmax_score_and_grad(h_final, alpha_f, M)
            
            h_final_new = h_final - lr_f * grad_f
            h_final_new = np.maximum(h_final_new, 0.0)
            
            score_f = compute_score(h_final_new)
            if score_f < best_score:
                best_score = score_f
                best_h = h_final_new.copy()
                h_final = h_final_new.copy()
            
            lr_f = max(lr_f * 0.99, 0.0001)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
