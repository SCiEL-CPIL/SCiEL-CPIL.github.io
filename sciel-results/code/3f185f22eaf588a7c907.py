"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h*h is linear autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h ** 2)
    
    # Compute linear autoconvolution using FFT
    n_conv = 2 * N - 1
    h_fft = np.fft.fft(h, n=2 * n_conv)
    conv = np.fft.ifft(h_fft * h_fft).real
    autoconv = conv[N-1:2*N-1]  # lags 0 to N-1
    autoconv_neg = conv[0:N]     # lags -(N-1) to 0 (reversed)
    autoconv_max = max(np.max(autoconv), np.max(autoconv_neg))
    
    # Ensure numerical stability
    autoconv_max = max(autoconv_max, 1e-15)
    
    ratio = l2_sq / (l1 * autoconv_max)
    return ratio


def spectral_to_heights(cos_coeffs):
    """
    Convert cosine coefficients to nonnegative heights via inverse Fourier.
    h[n] = sum_k cos_coeffs[k] * cos(2*pi*k*n/N) for k=0..K
    Then normalize to be nonnegative and L1=1.
    """
    n = np.arange(N)
    h = np.zeros(N)
    for k, c in enumerate(cos_coeffs):
        h += c * np.cos(2 * np.pi * k * n / N)
    # Ensure nonnegativity
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return np.ones(N) / N
    return h / l1


def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    Uses spectral parameterization with gradient ascent.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Initialize with uniform
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    # Number of cosine coefficients to use
    K = 8  # Use 9 coefficients (k=0 to 8)
    
    # Initialize cosine coefficients: start with a Gaussian-like spectrum
    def init_cos_coeffs(seed=0):
        rng = np.random.RandomState(seed)
        coeffs = np.zeros(K + 1)
        coeffs[0] = 1.0
        for k in range(1, K + 1):
            # Decay with frequency to create a smooth profile
            coeffs[k] = rng.randn() * np.exp(-k / 3.0)
        return coeffs
    
    # Try multiple initializations
    for seed in range(20):
        if time.time() - start_time > deadline - 30:
            break
        
        coeffs = init_cos_coeffs(seed)
        
        # Gradient ascent on cosine coefficients
        lr = 0.05
        momentum = 0.9
        v = np.zeros_like(coeffs)
        
        for it in range(500):
            if time.time() - start_time > deadline - 10:
                break
            
            h = spectral_to_heights(coeffs)
            current_ratio = compute_ratio(h)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = h.copy()
            
            # Compute gradient via finite differences
            grad = np.zeros_like(coeffs)
            eps = 1e-4
            for i in range(len(coeffs)):
                coeffs_plus = coeffs.copy()
                coeffs_plus[i] += eps
                h_plus = spectral_to_heights(coeffs_plus)
                r_plus = compute_ratio(h_plus)
                
                coeffs_minus = coeffs.copy()
                coeffs_minus[i] -= eps
                h_minus = spectral_to_heights(coeffs_minus)
                r_minus = compute_ratio(h_minus)
                
                grad[i] = (r_plus - r_minus) / (2 * eps)
            
            # Update with momentum
            v = momentum * v + grad
            coeffs = coeffs + lr * v
            
            # Keep coeffs[0] positive (DC component)
            coeffs[0] = max(coeffs[0], 0.1)
        
        # Final evaluation
        h_final = spectral_to_heights(coeffs)
        r_final = compute_ratio(h_final)
        if r_final > best_ratio:
            best_ratio = r_final
            best_h = h_final.copy()
    
    # Polish with Nelder-Mead on the full height vector
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        from scipy.optimize import minimize
        
        def neg_ratio(params):
            h = np.clip(params, 0, None)
            l1 = np.sum(h)
            if l1 < 1e-10:
                return 1e10
            h = h / l1
            return -compute_ratio(h)
        
        x0 = best_h.copy()
        try:
            result = minimize(neg_ratio, x0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < -best_ratio:
                best_h = np.clip(result.x, 0, None)
                best_ratio = -result.fun
        except:
            pass
    
    # Final check
    final_ratio = compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
    
    return list(best_h)


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
