import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

# Precompute FFT setup
# The step function has N intervals on [-1/4, 1/4], total width 0.5
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized score: max_conv / (integral_f)^2
    # For uniform heights=1: integral_f = 0.5, max_conv = 0.25
    # score = 0.25 / 0.25 = 1.0
    score = max_conv / (integral_f ** 2)
    
    return score

def compute_score_grad(heights):
    """Compute score and gradient using FFT-based approach."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return 1e10, np.zeros_like(heights)
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 1e-10:
        return 1e10, np.zeros_like(heights)
    
    score = max_conv / (integral_f ** 2)
    
    # Compute gradient
    # score = max_conv / I^2 where I = integral_f
    # d(score)/d(h_i) = (d(max_conv)/d(h_i) * I^2 - max_conv * 2*I * dI/d(h_i)) / I^4
    # = (d(max_conv)/d(h_i) / I^2) - 2*max_conv * dI/d(h_i) / I^3
    
    # dI/d(h_i) = w (since I = sum(h_j) * w)
    dI_dh = np.full(n, w)
    
    # d(max_conv)/d(h_i): The max is at some index in conv.
    # conv[k] = sum_j f_fine[j] * f_fine[(k-j) % M_FFT]
    # For circular convolution, if max is at index p:
    # d(conv[p])/d(f_fine[j]) = f_fine[(p-j) % M_FFT]
    # But we need d(conv[p])/d(h_i)
    # f_fine[j] = h_{j // K} for j in [0, M_STEP)
    # So d(f_fine[j])/d(h_i) = 1 if j // K == i and j < M_STEP, else 0
    
    # The peak of autoconvolution of a non-negative function is typically at lag 0 (index 0 or near M_FFT)
    # For circular convolution with zero padding, the "lag 0" term is at index 0.
    # Let's find the actual max index.
    peak_idx = np.argmax(conv)
    
    # Gradient of conv[peak_idx] w.r.t f_fine[j] is f_fine[(peak_idx - j) % M_FFT]
    # We need to map this back to h_i.
    # Let g_j = f_fine[(peak_idx - j) % M_FFT]
    # grad_h[i] = sum_{j: j//K == i, j < M_STEP} g_j
    
    # Efficient computation:
    # Create array G of length M_FFT where G[j] = f_fine[(peak_idx - j) % M_FFT]
    # This is just a roll of f_fine.
    G = np.roll(f_fine, peak_idx)
    
    # Now sum G over the indices corresponding to each h_i
    # Indices for h_i are [i*K, (i+1)*K)
    # We only care about j < M_STEP
    G_step = G[:M_STEP]
    # Reshape to (N, K) and sum along axis 1
    G_reshaped = G_step.reshape(N, K)
    grad_conv = np.sum(G_reshaped, axis=1)
    
    # Combine
    grad_score = grad_conv / (integral_f ** 2) - 2 * max_conv * dI_dh / (integral_f ** 3)
    
    return score, grad_score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy 1: Try various smooth shapes
    x = np.linspace(0, 1, N)
    
    # Raised cosine variations
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    # Plateau with tapered edges
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    # Gaussian-like shapes
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # Strategy 2: Random smooth perturbations
    rng = np.random.default_rng(42)
    for trial in range(50):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Create smooth perturbation using low-frequency Fourier modes
        g = np.zeros(N)
        num_modes = rng.integers(2, 10)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.3)
            g += amp * np.cos(2 * np.pi * k * x + phase)
        
        g -= np.mean(g)
        
        for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            if np.sum(h_trial) <= 0:
                continue
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Strategy 3: Gradient-based refinement
    if time.time() - START_TIME < TIME_LIMIT - 15:
        h_ref = best_h.copy()
        lr = 0.005
        for step in range(500):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            score, grad = compute_score_grad(h_ref)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            h_ref = h_ref - lr * grad
            h_ref = np.maximum(h_ref, 0.0)
            
            # Adaptive learning rate
            new_score = compute_score(h_ref)
            if new_score < best_score:
                best_score = new_score
                best_h = h_ref.copy()
                lr *= 1.05  # Slightly increase LR if improving
            else:
                h_ref = best_h.copy()  # Revert
                lr *= 0.9  # Decrease LR if not improving
    
    # Strategy 4: Final local search
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Try scaling the best shape
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            h_scaled = best_h * scale
            score = compute_score(h_scaled)
            if score < best_score:
                best_score = score
                best_h = h_scaled.copy()
        
        # Try local search around best
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.01, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Final gradient polish
    if time.time() - START_TIME < TIME_LIMIT - 2:
        h_final = best_h.copy()
        lr = 0.001
        for step in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 1:
                break
            
            score, grad = compute_score_grad(h_final)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            h_final = h_final - lr * grad
            h_final = np.maximum(h_final, 0.0)
            
            new_score = compute_score(h_final)
            if new_score < best_score:
                best_score = new_score
                best_h = h_final.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
