import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s

# Precompute FFT setup
# The step function has N intervals on [-1/4, 1/4], total width 0.5
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

# Precompute the index mapping for fine grid
# Each step interval i maps to indices [i*K, (i+1)*K) in the fine grid
# We'll build the fine grid by repeating each height K times
def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    # Use repeat for efficiency
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Build fine-sampled function
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized max: max_conv * support_length / (integral_f)^2
    # For uniform heights=1: integral_f = 0.5, max_conv = 0.25
    # score = 0.25 * 0.5 / 0.25 = 0.5 ... but we want 1.0 for uniform
    # Let's use: max_conv / (integral_f)^2
    # For uniform: 0.25 / 0.25 = 1.0 ✓
    score = max_conv / (integral_f ** 2)
    
    return score

def compute_score_fast(heights):
    """Faster version using precomputed structures for optimization."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return 1e10
    heights = np.maximum(heights, 0.0)
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f <= 1e-10:
        return 1e10
    
    return max_conv / (integral_f ** 2)

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy: Try various smooth shapes and refine
    # 1. Raised cosine
    x = np.linspace(0, 1, N)
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    # 2. Plateau with tapered edges
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
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
    
    # 3. Gaussian-like shape
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # 4. Random smooth perturbations around uniform
    rng = np.random.default_rng(42)
    for trial in range(30):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Create smooth perturbation using low-frequency Fourier modes
        g = np.zeros(N)
        num_modes = rng.integers(2, 8)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.2)
            g += amp * np.cos(2 * np.pi * k * x + phase)
        
        g -= np.mean(g)
        
        for eps in [0.05, 0.1, 0.15, 0.2, 0.3]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            if np.sum(h_trial) <= 0:
                continue
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # 5. Gradient-based refinement from best shape
    if time.time() - START_TIME < TIME_LIMIT - 15:
        h_ref = best_h.copy()
        lr = 0.001
        for step in range(300):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Compute gradient using FFT
            f_fine = np.zeros(M_FFT, dtype=np.float64)
            f_fine[:M_STEP] = np.repeat(h_ref, K)
            F = np.fft.rfft(f_fine)
            conv = np.fft.irfft(F * F, n=M_FFT)
            
            peak_idx = np.argmax(conv)
            # Subgradient: direction to reduce peak
            # The peak of autoconvolution is at lag 0 (peak_idx should be 0 or near end)
            # For circular convolution, peak at 0
            # Gradient of max_conv w.r.t. h_i involves h at positions that contribute to peak
            # Simple approach: reduce heights where they contribute most to peak
            contrib = np.roll(h_ref, peak_idx % N)
            grad = 2.0 * contrib / (np.sum(h_ref) + 1e-10)
            
            h_ref = h_ref - lr * grad
            h_ref = np.maximum(h_ref, 0.0)
            
            score = compute_score_fast(h_ref)
            if score < best_score:
                best_score = score
                best_h = h_ref.copy()
    
    # Final: try a few more refinements
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Try scaling the best shape
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            h_scaled = best_h * scale
            score = compute_score(h_scaled)
            if score < best_score:
                best_score = score
                best_h = h_scaled.copy()
        
        # Try local search around best
        rng2 = np.random.default_rng(123)
        for _ in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.01, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
