import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0

def compute_ratio(h, n_grid=2000):
    """Compute max(autoconv) / l2_sq for step function h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]

    # Linear autoconvolution via FFT (zero-padded)
    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)

    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _compute_ratio_from_fourier(params, n_fine=1024, K=25):
    """
    Reconstruct function from Fourier coefficients, clip to non-negative,
    normalize L2 norm to 1, and return the peak of the autoconvolution.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    # Unpack parameters
    a0 = params[0]
    a = params[1::2]
    b = params[2::2]
    
    # Ensure correct lengths
    K_actual = len(a)
    if K_actual < K:
        # Pad if necessary (shouldn't happen if fixed size)
        pad_a = np.zeros(K - K_actual)
        pad_b = np.zeros(K - K_actual)
        a = np.concatenate([a, pad_a])
        b = np.concatenate([b, pad_b])
    else:
        a = a[:K]
        b = b[:K]

    # Grid
    x = np.linspace(-0.25, 0.25, n_fine, endpoint=False)
    L = 0.5
    # Frequencies
    freqs = np.arange(K)
    
    # Compute f(x)
    f = np.full(n_fine, a0)
    for k in range(1, K):
        f += a[k-1] * np.cos(2 * np.pi * k * x / L)
        f += b[k-1] * np.sin(2 * np.pi * k * x / L)
        
    # Clip to non-negative
    f = np.clip(f, 0.0, None)
    
    # Normalize L2 norm to 1
    l2_sq = np.sum(f**2) * (L / n_fine)
    if l2_sq < 1e-10:
        return 1e6 # Penalty for zero function
    f = f / np.sqrt(l2_sq)
    
    # Compute autoconvolution peak
    # Use FFT for efficiency
    n_fft = 1 << (2 * n_fine).bit_length()
    F = np.fft.rfft(f, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_fine - 1]
    dx = L / n_fine
    conv *= dx
    max_conv = np.max(conv)
    
    return max_conv

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    K = 25 # Number of harmonics
    # Parameter vector size: a0, (a1, b1), ..., (aK-1, bK-1) -> 1 + 2*(K-1)
    n_params = 1 + 2 * (K - 1)
    
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Initial guess: Gaussian-like shape in Fourier domain?
    # Or just random small values.
    
    restarts = 0
    while time.time() < deadline and restarts < 10:
        restarts += 1
        
        # Initialize parameters
        # Start with a dominant constant term and small noise
        x0 = np.zeros(n_params)
        x0[0] = 1.0 # Constant term
        # Add small random noise to break symmetry
        noise = rng.standard_normal(n_params - 1) * 0.1
        x0[1:] = noise
        
        # Run Nelder-Mead optimization
        # n_fine=512 is sufficient for estimating the peak during optimization
        # We use a higher resolution in the final evaluation
        res = opt.minimize(
            _compute_ratio_from_fourier, 
            x0, 
            args=(512, K), 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6}
        )
        
        if res.fun < best_ratio:
            best_ratio = res.fun
            # Reconstruct the function on the 600-step grid from the optimized coefficients
            params = res.x
            a0 = params[0]
            a = params[1::2][:K-1]
            b = params[2::2][:K-1]
            
            # Evaluate on the actual 600-step grid
            x_step = np.linspace(-0.25, 0.25, N, endpoint=False)
            L = 0.5
            h_new = np.full(N, a0)
            for k in range(1, K):
                h_new += a[k-1] * np.cos(2 * np.pi * k * x_step / L)
                h_new += b[k-1] * np.sin(2 * np.pi * k * x_step / L)
            
            h_new = np.clip(h_new, 0.0, None)
            
            # Verify the ratio on the discrete step function
            r_discrete = compute_ratio(h_new)
            if r_discrete < best_ratio:
                best_ratio = r_discrete
                best_h = h_new.copy()
            else:
                # The continuous approximation was better, but the discrete version is worse?
                # This can happen due to aliasing or the step approximation.
                # We stick with the best discrete result found so far.
                pass

    return best_h.tolist()
