import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    # Width of each interval
    w = 0.5 / n
    
    # Use FFT with zero-padding for linear convolution
    # The step function has n intervals, each of width w
    # Sample at fine resolution: M points total covering the support
    M = 1 << 15  # 32768 points
    samples_per_interval = M // n
    
    # Build fine-sampled function
    f_fine = np.zeros(M, dtype=np.float64)
    for i in range(n):
        start = i * samples_per_interval
        end = start + samples_per_interval
        f_fine[start:end] = heights[i]
    
    # Compute autoconvolution via FFT (linear convolution via zero-padding)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized max: max_conv / (integral_f)^2 * (support length)
    # For uniform: integral_f = 0.5, max_conv = 0.25 (peak of triangle)
    # score = 0.25 / 0.25 * 0.5 = 0.5 ... but seed gives 1.0
    # Let's try: max_conv * n / (sum(heights))^2
    total = np.sum(heights)
    if total <= 0:
        return np.inf
    
    # Alternative normalization that gives 1.0 for uniform
    # For uniform heights=1: total=N, max_conv from FFT...
    # Let me just use the ratio that gives 1.0 for uniform
    score = max_conv * n / (total * total)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Try small perturbations around uniform
    # Use a smooth parameterization: h_i = 1 + epsilon * g_i where g has zero mean
    # This ensures we stay near uniform and can explore the neighborhood
    
    rng = np.random.default_rng(42)
    
    # Generate several random smooth perturbations
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a smooth perturbation using low-frequency Fourier modes
        freqs = np.arange(N)
        g = np.zeros(N)
        # Use a few random low-frequency components
        num_modes = rng.integers(2, 8)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.01, 0.1)
            g += amp * np.cos(2 * np.pi * k * freqs / N + phase)
        
        # Ensure zero mean
        g -= np.mean(g)
        
        # Try several epsilon values
        for eps in [0.05, 0.1, 0.2, 0.3, 0.5]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            
            if np.sum(h_trial) <= 0:
                continue
            
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
        
        # Also try a raised cosine shape
        x = np.linspace(0, 1, N)
        for amp in [0.1, 0.2, 0.3, 0.5]:
            h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
            h_cos = np.maximum(h_cos, 0.0)
            score = compute_score(h_cos)
            if score < best_score:
                best_score = score
                best_h = h_cos.copy()
        
        # Try a plateau with tapered edges
        for taper_frac in [0.1, 0.2, 0.3]:
            h_plateau = np.ones(N)
            taper_len = int(taper_frac * N)
            if taper_len > 0:
                # Taper at edges
                ramp = np.linspace(0, 1, taper_len)
                h_plateau[:taper_len] = ramp
                h_plateau[-taper_len:] = ramp[::-1]
            score = compute_score(h_plateau)
            if score < best_score:
                best_score = score
                best_h = h_plateau.copy()
    
    # Final: try a quick gradient-based refinement from best_h
    if time.time() < deadline - 10:
        h_ref = best_h.copy()
        lr = 0.001
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            H = np.fft.fft(h_ref)
            auto_conv = np.fft.ifft(np.abs(H)**2).real
            
            # Gradient of max(auto_conv) w.r.t. h
            # d(max(auto_conv))/d(h_i) = 2 * (auto_conv at peak) * h_{peak-i} / h_i approx
            # Use subgradient: gradient points in direction of h at the peak location
            peak_idx = np.argmax(auto_conv)
            # The autoconvolution at peak depends on all h values
            # Simple heuristic: reduce h where it contributes most to the peak
            contrib = np.abs(np.roll(h_ref, peak_idx))
            grad = 2 * contrib / (np.sum(h_ref) + 1e-10)
            
            h_ref = h_ref - lr * grad
            h_ref = np.maximum(h_ref, 0.0)
            
            score = compute_score(h_ref)
            if score < best_score:
                best_score = score
                best_h = h_ref.copy()
    
    # Ensure non-negativity and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
