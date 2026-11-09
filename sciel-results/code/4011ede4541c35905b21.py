import numpy as np
import time
from scipy import optimize

N = 600
L = 0.5
dx = L / N

def compute_autoconv_max(h):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    Uses FFT-based convolution for accuracy.
    """
    if np.any(h < -1e-12):
        return float('inf')
    h = np.clip(h, 0, None)
    
    # Compute autoconvolution using FFT
    # The autoconvolution g(t) = integral f(x) f(t-x) dx
    # For step functions, we can compute this via FFT
    n_fft = 2 * N
    f_ext = np.zeros(n_fft)
    f_ext[:N] = h
    
    # Convolve f with f
    F = np.fft.rfft(f_ext)
    conv = np.fft.irfft(F * F, n=n_fft)
    
    # The convolution gives us the integral of f(x)f(y) dx dy over overlapping regions
    # But we need to be careful about the scaling
    # Actually, for discrete convolution: (f * f)[k] = sum_i f[i] f[k-i]
    # This is an approximation to the continuous convolution
    
    # The continuous autoconvolution at t=0 is sum(h_i^2) * dx
    # Let's compute the max of the discrete convolution and scale appropriately
    
    # The discrete convolution approximates (1/dx) * continuous_conv
    # So continuous_conv ≈ discrete_conv * dx
    
    # Let's compute the max of the autoconvolution
    # The peak should be at t=0 for symmetric unimodal functions
    # But let's find the actual max
    
    # The convolution result has length 2*N-1 conceptually, but with FFT we get n_fft
    # The autoconvolution is symmetric, so we look at the center
    # The peak is at index N-1 (corresponding to t=0)
    
    # Actually, let me think more carefully.
    # f_ext = [h_0, h_1, ..., h_{N-1}, 0, ..., 0]
    # conv[k] = sum_i f_ext[i] * f_ext[k-i]
    # For k = N-1: conv[N-1] = sum_i f_ext[i] * f_ext[N-1-i]
    # This includes terms where both indices are in [0, N-1]
    
    # The continuous autoconvolution g(t) = integral f(x) f(x-t) dx
    # For step functions, this is a piecewise quadratic function
    # The maximum is typically at t=0 for non-negative symmetric functions
    
    # Let's compute g(0) = sum(h_i^2) * dx
    g0 = np.sum(h**2) * dx
    
    # But the max might not be at t=0 for all functions
    # Let's compute the full autoconvolution on a fine grid
    
    # Alternative: use the fact that for non-negative functions, the autoconvolution
    # is maximized at t=0 if the function is "peaky" or at the edges if spread out
    
    # For safety, let's compute the max using FFT-based convolution
    # The discrete auto-correlation c[k] = sum_i h[i] h[i+k] for k >= 0
    # This is related to the continuous autoconvolution
    
    # Let's use a more direct approach: compute the autoconvolution at multiple points
    # using the FFT result
    
    # The FFT convolution gives us values at discrete points
    # conv[k] ≈ (1/dx) * g(k*dx) for appropriate k
    
    # Let's find the max of conv and scale
    # The autoconvolution is symmetric, so we look at conv[0:N]
    conv_max = np.max(conv[:N])
    
    # Scale: continuous g(t) ≈ conv[k] * dx
    # But we need to verify the scaling
    
    # For uniform h=1: g(0) = L = 0.5
    # conv[0] for uniform = N (sum of N ones times ones)
    # So g(0) ≈ N * dx = 0.5. That works!
    
    # So the normalized max = max(g) / (integral)^2
    I = np.sum(h) * dx
    if I == 0:
        return float('inf')
    
    max_g = conv_max * dx
    return max_g / (I**2)

def spectral_to_heights(amps, phases=None):
    """
    Convert spectral amplitudes to time-domain heights.
    amps: array of length N//2+1 (non-negative)
    Returns: array of length N (heights)
    """
    if phases is None:
        # Use zero phase (symmetric function)
        F = np.zeros(N // 2 + 1, dtype=complex)
        F = amps.astype(complex)
    else:
        F = amps * np.exp(1j * phases)
    
    # Inverse FFT to get time domain
    h = np.fft.irfft(F, n=N)
    
    # Ensure non-negativity using softplus or ReLU
    # Use softplus for smoothness: h = log(1 + exp(x))
    # But this shifts the function. Let's use ReLU for simplicity
    h = np.clip(h, 0, None)
    
    return h

def heights_to_spectral(h):
    """
    Convert heights to spectral amplitudes.
    """
    F = np.fft.rfft(h)
    amps = np.abs(F)
    return amps

def objective(x):
    """
    Objective function for optimization.
    x: spectral amplitudes (length N//2+1)
    """
    h = spectral_to_heights(x)
    return compute_autoconv_max(h)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialize with uniform function
    h_init = np.ones(N)
    amps_init = heights_to_spectral(h_init)
    
    # Also try a few other initializations
    candidates = []
    
    # Uniform
    candidates.append(amps_init.copy())
    
    # Triangle
    h_tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        dist = abs(i - center)
        if dist < center:
            h_tri[i] = 1.0 - (dist / center)
    candidates.append(heights_to_spectral(h_tri))
    
    # Gaussian-like
    x_grid = np.linspace(-L/2, L/2, N)
    h_gauss = np.exp(-10 * x_grid**2)
    h_gauss = h_gauss / np.max(h_gauss)
    candidates.append(heights_to_spectral(h_gauss))
    
    best_h = h_init
    best_score = float('inf')
    
    for amps in candidates:
        # Run gradient descent from this initialization
        try:
            result = optimize.minimize(
                objective,
                amps,
                method='L-BFGS-B',
                bounds=[(0, None)] * len(amps),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
                
            h_opt = spectral_to_heights(result.x)
            score = compute_autoconv_max(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception as e:
            pass
    
    # Also evaluate the original candidates
    for amps in candidates:
        h = spectral_to_heights(amps)
        score = compute_autoconv_max(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    return list(best_h)
