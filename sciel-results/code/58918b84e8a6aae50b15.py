import numpy as np
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute autoconvolution via FFT for speed."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
    return conv

def spectral_objective(mag, beta=20.0):
    """
    Objective in frequency domain: minimize smooth max of autoconv.
    mag: real nonnegative array of length N//2+1 (rfft magnitude)
    We reconstruct f via softplus to ensure nonnegativity, then compute
    a smooth surrogate of max(autoconv) / (sum(f))^2.
    """
    # Reconstruct f from magnitude spectrum (phase is zero => even function)
    # Actually, to allow non-even, we need phase. But for simplicity and symmetry,
    # we start with zero phase (even function) and later refine.
    # For now, use zero phase: f = irfft(mag * exp(0j))
    n_fft = 1 << (2 * N - 1).bit_length()
    F = mag.astype(np.complex128)
    # Pad to full rfft length
    F_full = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    F_full[:len(mag)] = F
    # Ensure conjugate symmetry is handled by irfft
    f = np.fft.irfft(F_full, n=n_fft)[:N]
    
    # Softplus projection to ensure nonnegativity
    # f = softplus(f) is not quite right; we want f >= 0.
    # Better: define f = exp(log_mag) but we already have mag.
    # Actually, let's parameterize f directly as nonnegative via softplus of a latent variable.
    # But here we are in spectral domain. Let's stick to time-domain parameterization
    # but use FFT for fast convolution in the objective.
    # Wait, the directive says "different representation". 
    # Let's try: parameterize f as nonnegative, but optimize using FFT-based gradients.
    # Actually, a true spectral approach: minimize max_k |F(k)|^2? No, max of time-domain conv.
    
    # Let's revert to a robust time-domain optimization but using FFT for speed
    # and a different initialization: random smooth functions.
    pass

# Since a pure spectral magnitude optimization with fixed phase is too restrictive,
# and the "inspiration" mentions spectral coordinate descent, let's implement:
# 1. Parameterize f as nonnegative via softplus(z)
# 2. Use FFT for fast autoconvolution
# 3. Use a smooth max surrogate
# 4. Use coordinate descent or Nelder-Mead on z (latent variables)

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))

def f_from_z(z):
    f = softplus(z)
    return f

def objective_z(z, beta=20.0):
    f = f_from_z(z)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    conv = fft_autoconv(f)
    # Smooth max
    c_max = np.max(conv)
    if c_max < 1e-15:
        return 0.0
    shifted = conv / c_max
    log_sum = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    smooth_max_val = c_max * (1.0 + log_sum)
    return smooth_max_val / (s * s)

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Latent variable z such that f = softplus(z)
    # For constant f=1, z = log(exp(1)-1) ~ 0.54
    z0 = np.log(np.exp(1.0) - 1.0) * np.ones(N)
    
    rng = np.random.RandomState(42)
    best_z = z0.copy()
    best_obj = objective_z(z0)
    
    # Multiple restarts with different initializations
    for restart in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Random smooth initialization
        if restart == 0:
            z = z0.copy()
        else:
            # Generate random smooth function via low-frequency Fourier coefficients
            freqs = np.fft.rfftfreq(N)
            # Only low frequencies
            max_freq = N // 10
            low_mask = freqs <= max_freq
            coeffs = rng.randn(N // 2 + 1) * low_mask
            coeffs[0] = 0.5  # mean
            f_init = np.fft.irfft(coeffs * N, n=N)
            # Ensure nonnegative by shifting
            f_init = np.maximum(f_init + np.abs(f_init).min() + 0.1, 0.1)
            # Convert to z
            z = np.log(np.expm1(f_init))
        
        # Optimize z using Nelder-Mead
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_z, z, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_z = result.x
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    # Convert best_z to f
    best_f = f_from_z(best_z)
    best_score = compute_score(best_f)
    
    # Local refinement: coordinate descent on f directly (nonnegative)
    if time.time() - start_time < TIME_LIMIT - 5:
        current_f = best_f.copy()
        current_score = best_score
        alpha = 0.01
        for step in range(500):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            # Compute gradient-like sensitivity
            conv = fft_autoconv(current_f)
            s = np.sum(current_f)
            c_max = np.max(conv)
            current_score = c_max / (s * s)
            
            # Find peak lags
            threshold = c_max * 0.95
            peak_indices = np.where(conv >= threshold)[0]
            
            # Sensitivity: d(conv[m])/d(f[j]) = f[m-j]
            sensitivity = np.zeros(N)
            for m in peak_indices:
                j_min = max(0, m - (N - 1))
                j_max = min(N - 1, m)
                for j in range(j_min, j_max + 1):
                    sensitivity[j] += current_f[m - j]
            
            # Normalize
            sens_mean = np.mean(sensitivity)
            sens_std = np.std(sensitivity) + 1e-12
            sens_norm = (sensitivity - sens_mean) / sens_std
            
            # Update
            f_new = current_f * (1.0 - alpha * sens_norm)
            f_new = np.maximum(f_new, 0.0)
            
            # Normalize to keep sum constant
            s_new = np.sum(f_new)
            if s_new < 1e-12:
                break
            f_new = f_new * (s / s_new)
            
            new_score = compute_score(f_new)
            if new_score < current_score:
                current_f = f_new
                current_score = new_score
                alpha = min(alpha * 1.02, 0.1)
            else:
                alpha *= 0.9
                if alpha < 1e-6:
                    break
        
        if current_score < best_score:
            best_f = current_f
            best_score = current_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
