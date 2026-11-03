"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Fourier-domain spectral shaping. The autoconvolution's peak is
controlled by the spectral content. We initialize with a flat-spectrum (sinc-like)
shape, then use a proximal gradient method that works in Fourier space to
flatten the spectral magnitude, which directly reduces the autoconvolution peak.
We combine this with a final time-domain sensitivity refinement.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

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

def fft_conv(f):
    """Compute autoconvolution via FFT for speed."""
    n = 2 * N
    F = np.fft.rfft(f, n=n)
    conv = np.fft.irfft(F * F, n=n)[:2*N-1]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x / x_max - 1.0
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def make_sinc_init(n_bumps=1, width_frac=0.8):
    """Create a sinc-like initialization with flat spectral content.
    
    A function with a rectangular spectrum has a sinc time-domain shape.
    We create a sum of shifted sincs to get nonnegative values.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Single sinc-like bump (flat spectrum in a band)
    w = width_frac * 0.5  # half-width
    f = np.zeros(N)
    for i, xi in enumerate(x):
        t = xi / w
        if abs(t) < 1e-10:
            f[i] = 1.0
        else:
            f[i] = np.sinc(t)  # sin(pi*t)/(pi*t)
    
    # Make nonnegative by taking absolute value and shifting
    f = np.abs(f)
    # Add a small constant to ensure nonnegativity and smoothness
    f = f + 0.1
    return f

def make_flat_spectrum_init():
    """Initialize with a function whose DFT magnitude is as flat as possible.
    
    We start with a constant (flat spectrum) and modulate to reduce peak.
    """
    # Start with a raised cosine window (good spectral properties)
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    f = 0.5 + 0.5 * np.cos(x)
    return f

def make_bump_init(center=0.0, width=0.4):
    """A single smooth bump."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f = np.exp(-((x - center) / width) ** 2)
    return f

def spectral_proximal_gradient(f0, start_time, max_time, beta=20.0):
    """Proximal gradient method minimizing smooth_max(conv) / (sum(f))^2.
    
    Works in Fourier space for efficient gradient computation.
    """
    f = f0.copy().astype(np.float64)
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return f0, compute_score(f0)
    
    # Normalize
    f = f / s * N
    s = np.sum(f)
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Precompute FFT sizes
    n_fft = 2 * N
    freqs = np.fft.rfftfreq(n_fft, d=1.0/n_fft)
    
    # Adaptive step size
    step = 1.0 / (2 * N)
    momentum = 0.9
    y = f.copy()
    t_k = 1.0
    
    start = time.time()
    while time.time() - start < max_time:
        # Compute FFT
        F = np.fft.rfft(f, n=n_fft)
        # Autoconvolution in frequency domain: |F|^2
        # Inverse FFT gives convolution
        conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
        
        s = np.sum(f)
        if s < 1e-15:
            break
        
        # Objective value
        val = np.max(conv) / (s * s)
        if val < best_val:
            best_val = val
            best_f = f.copy()
        
        # Gradient of smooth_max(conv, beta) / s^2 with respect to f
        # d/d f_j [conv[m]] = f[m-j] (for valid m)
        # Using smooth max: gradient of smooth_max is softmax-weighted
        
        x_max = np.max(conv)
        if x_max < 1e-15:
            grad_conv = np.zeros_like(conv)
        else:
            shifted = conv / x_max - 1.0
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            grad_conv = weights  # gradient of smooth_max w.r.t. conv
        
        # Gradient of conv w.r.t. f
        # conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/d(f[j]) = f[m-j]
        # So grad_f[j] = sum_m grad_conv[m] * f[m-j] / s^2 - 2 * conv[j] * f[j] / s^3 * s
        # More carefully:
        # obj = max(conv) / s^2
        # d(obj)/d(f[j]) = (sum_m grad_conv[m] * d(conv[m])/d(f[j])) / s^2 - 2 * max(conv) * f[j] / s^3
        # d(conv[m])/d(f[j]) = f[m-j] (where m-j is in range)
        
        # Compute sum_m grad_conv[m] * f[m-j] for each j
        # This is a cross-correlation of grad_conv with f
        # grad_f_conv[j] = sum_m grad_conv[m] * f[m-j]
        # = (grad_conv * f)[j] in convolution sense
        
        # Use FFT for this
        G = np.fft.rfft(grad_conv, n=n_fft)
        # cross_corr[j] = sum_m grad_conv[m] * f[m-j]
        # = ifft(fft(grad_conv) * conj(fft(f)))[j]
        cross = np.fft.irfft(G * np.conj(F), n=n_fft)[:N]
        
        # Gradient of obj w.r.t. f
        grad_obj = cross / (s * s) - 2.0 * x_max * f / (s * s * s)
        
        # Proximal gradient step with momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        y = f + (t_k / t_new) * (y - f)
        y = y - step * grad_obj
        # Project onto nonnegativity
        y = np.maximum(y, 0.0)
        # Normalize
        s_y = np.sum(y)
        if s_y < 1e-15:
            break
        y = y / s_y * N
        
        f = y.copy()
        t_k = t_new
        
        # Adaptive step size
        if val < best_val:
            step = min(step * 1.02, 1.0)
        else:
            step *= 0.95
        
        if step < 1e-8:
            break
    
    return best_f, best_val

def sensitivity_refinement(f, start_time, max_time):
    """Final sensitivity-based refinement in time domain."""
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    alpha = 0.03
    threshold_frac = 0.97
    
    start = time.time()
    while time.time() - start < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_conv(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find peak lags
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            break
        
        # Compute sensitivity: for each bin j, contribution to peak lags
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_max >= j_min:
                sensitivity[j_min:j_max+1] += current_f[m - j_min:m - j_min + (j_max - j_min + 1)]
        
        # Normalize
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce high-sensitivity bins
        h_new = current_f * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_conv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    
    # 1. Sinc-like (flat spectrum)
    inits.append(make_sinc_init())
    
    # 2. Raised cosine
    inits.append(make_flat_spectrum_init())
    
    # 3. Gaussian bump
    inits.append(make_bump_init(center=0.0, width=0.15))
    
    # 4. Wider Gaussian
    inits.append(make_bump_init(center=0.0, width=0.25))
    
    # 5. Constant
    inits.append(np.ones(N))
    
    # 6. Two-bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_two = np.exp(-((x + 0.1)/0.12)**2) + np.exp(-((x - 0.1)/0.12)**2)
    inits.append(f_two)
    
    # 7. Plateau with smooth edges
    f_plateau = np.ones(N)
    edge = N // 10
    for i in range(edge):
        t = i / edge
        f_plateau[i] = 0.5 * (1 + np.cos(np.pi * t))
        f_plateau[N-1-i] = 0.5 * (1 + np.cos(np.pi * t))
    inits.append(f_plateau)
    
    # Evaluate all inits
    best_f = None
    best_score = np.inf
    for f_init in inits:
        f_init = np.maximum(f_init, 0.0)
        sc = compute_score(f_init)
        if sc < best_score:
            best_score = sc
            best_f = f_init.copy()
    
    if best_f is None:
        best_f = np.ones(N)
        best_score = compute_score(best_f)
    
    # Spectral proximal gradient refinement on best init
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = spectral_proximal_gradient(
            best_f, start_time, TIME_LIMIT - 25, beta=15.0
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Also try spectral PG on other good inits
    for f_init in inits[:4]:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        f_init = np.maximum(f_init, 0.0)
        if compute_score(f_init) < best_score * 1.1:  # only if competitive
            refined_f, refined_score = spectral_proximal_gradient(
                f_init, start_time, TIME_LIMIT - 20, beta=15.0
            )
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score
    
    # Final sensitivity refinement
    if time.time() - start_time < TIME_LIMIT - 5:
        final_f, final_score = sensitivity_refinement(
            best_f, start_time, TIME_LIMIT - 3
        )
        if final_score < best_score:
            best_f = final_f
            best_score = final_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
