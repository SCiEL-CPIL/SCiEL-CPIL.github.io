import numpy as np
import time
import scipy.optimize as opt

N = 600
# Precompute FFT size for linear convolution of length N with itself
# Result length is 2*N - 1. We need an FFT size >= 2*N - 1.
FFT_SIZE = 1 << (2*N - 1).bit_length()

# Precompute a fine grid for accurate evaluation of the peak
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to interval indices
W = 0.5 / N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: max(g) / sum(h^2)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # L2 norm squared
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
        
    # Expand h to fine grid
    f = h[CELL]
    
    # FFT convolution
    F = np.fft.rfft(f, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1] * DX
    
    peak = np.max(conv)
    return float(peak / l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Phase 1: Initialize with a few candidate shapes
    # Uniform
    h_uniform = np.ones(N)
    # Gaussian centered
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h_gauss = np.exp(-20 * x**2)
    # Cosine bump (positive on [-1/4, 1/4])
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x))  # peaks at 0, zero at ends
    
    best_h = h_uniform.copy()
    best_val = ratio_from_h(h_uniform)
    
    for h_init in [h_gauss, h_cos]:
        v = ratio_from_h(h_init)
        if v < best_val:
            best_val = v
            best_h = h_init.copy()
    
    # Phase 2: Low-dimensional Fourier representation optimization
    # We represent h(x) = a_0 + sum_{k=1}^{K} a_k * cos(2*pi*k*x)
    # This enforces smoothness and reduces the dimensionality from 600 to K+1.
    # We optimize the coefficients to minimize the objective.
    
    K = 30  # Number of harmonics
    # Precompute basis matrix B where B[i, j] is the value of basis function j at interval i
    # Basis functions: 1, cos(2*pi*1*x), ..., cos(2*pi*K*x)
    # Note: x is the center of the interval or just the grid point? 
    # The step function is constant on intervals. The value is h[i].
    # The continuous function f(t) for t in interval i is h[i].
    # The autoconvolution of the step function corresponds to the convolution of the sequence h.
    # However, the "normalized maximum" metric in the task usually refers to the continuous autoconvolution 
    # or the discrete one? 
    # The seed program uses a fine grid expansion to approximate the continuous convolution.
    # If we use a low-frequency cosine series, the function is smooth.
    # Let's define the coefficients such that h[i] = sum_{k=0}^K c_k * cos(2*pi*k*x_i)
    # where x_i is the center of the i-th interval.
    
    x_centers = np.linspace(-0.25 + W/2, 0.25 - W/2, N)
    B = np.zeros((N, K + 1))
    B[:, 0] = 1.0
    for k in range(1, K + 1):
        B[:, k] = np.cos(2 * np.pi * k * x_centers)
    
    # Objective for coefficients c
    def obj_coeffs(c):
        h = B @ c
        # Ensure non-negativity? The cosine series might go negative.
        # We can clip it, but that breaks the smoothness assumption slightly.
        # Or we can constrain the optimization to keep it positive.
        # For now, clip and evaluate.
        h_clipped = np.clip(h, 0.0, None)
        return ratio_from_h(h_clipped)
    
    # Initial guess: Uniform (only c_0 non-zero)
    # For uniform h=c, B @ c gives c * [1, 0, 0...]? No.
    # B[:, 0] is 1. So c_0 * 1 + 0 = c_0.
    # So c_0 = 1.0, others 0.
    c_init = np.zeros(K + 1)
    c_init[0] = 1.0
    
    # Try to optimize using Nelder-Mead
    # Since K=30, this is a 31-dimensional problem. NM might be slow but doable with limits.
    # Alternatively, use L-BFGS-B if we can provide a gradient? 
    # The objective is non-smooth due to max and clip. 
    # Let's try NM with a limited number of iterations.
    
    best_c = c_init.copy()
    best_c_val = obj_coeffs(c_init)
    
    # Multi-start NM
    rng = np.random.default_rng(42)
    num_starts = 5
    
    for s in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        
        if s == 0:
            c_cur = c_init.copy()
        else:
            # Random perturbation of coefficients
            c_cur = c_init.copy()
            c_cur[1:] += 0.1 * rng.normal(0, 1, size=K)
        
        # Limit iterations for NM
        maxiter = 500
        res = opt.minimize(obj_coeffs, c_cur, method='Nelder-Mead', 
                           options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_c_val:
            best_c_val = res.fun
            best_c = res.x
    
    # Convert best coefficients to h
    h_fourier = B @ best_c
    h_fourier = np.clip(h_fourier, 0.0, None)
    
    v_fourier = ratio_from_h(h_fourier)
    if v_fourier < best_val:
        best_val = v_fourier
        best_h = h_fourier.copy()
    
    # Phase 3: Local refinement on the full 600-dim vector
    # Use the best h from Phase 2 as the starting point for local search.
    # This combines the global structure of the Fourier fit with local adjustments.
    
    current_h = best_h.copy()
    current_val = best_val
    
    max_iters = 3000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random block of coordinates to perturb
        block_size = int(rng.integers(1, 10))
        start_idx = int(rng.integers(0, N - block_size + 1))
        end_idx = start_idx + block_size
        
        # Perturb the block
        h_try = current_h.copy()
        # Small random additive noise
        noise = rng.normal(0, 0.01, size=block_size)
        h_try[start_idx:end_idx] += noise
        h_try = np.clip(h_try, 0.0, None)
        
        v = ratio_from_h(h_try)
        if v < current_val:
            current_val = v
            current_h = h_try
            if current_val < best_val:
                best_val = current_val
                best_h = h_try.copy()
    
    return list(best_h)
