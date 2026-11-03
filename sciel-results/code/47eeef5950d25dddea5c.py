"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize_scalar, minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin

    # 1. Parametric Window Optimization
    # We assume the optimal shape is positive and smooth.
    # We will optimize a few families of windows.
    
    best_val = float('inf')
    best_h = None
    
    # Family 1: All Ones
    h_ones = np.ones(N)
    val_ones = _score(h_ones)
    if val_ones < best_val:
        best_val = val_ones
        best_h = h_ones.copy()

    # Family 2: Hanning-like
    # h_i = A + B * cos(2*pi*i/(N-1))
    # Standard Hanning: 0.5 + 0.5*cos(...) -> zeros at ends.
    # We can scale it. Scale doesn't matter.
    def make_hanning(alpha):
        # alpha controls the "flatness" or taper
        # alpha=1 is standard Hanning (0 at ends)
        # alpha=0 is rectangular
        # h_i = 1 - alpha * 0.5 * (1 - cos(2*pi*i/(N-1)))
        # Or simply: 0.5*(1+alpha) + 0.5*(1-alpha)*cos(...)
        i = np.arange(N)
        if N > 1:
            phase = 2 * np.pi * i / (N - 1)
        else:
            phase = 0
        h = 0.5 * (1 + alpha) + 0.5 * (1 - alpha) * np.cos(phase)
        return h

    res_hann = minimize_scalar(lambda a: _score(make_hanning(a)), bounds=(0, 1), method='bounded')
    if res_hann.success:
        h_opt = make_hanning(res_hann.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Family 3: Gaussian
    def make_gaussian(sigma):
        i = np.arange(N)
        center = (N - 1) / 2.0
        h = np.exp(-((i - center) ** 2) / (2 * sigma * sigma))
        return h

    res_gauss = minimize_scalar(lambda s: _score(make_gaussian(s)), bounds=(1, N), method='bounded')
    if res_gauss.success:
        h_opt = make_gaussian(res_gauss.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Family 4: Power Law (Waterfall)
    # h_i = (i+1)^p or similar? No, symmetric is better.
    # h_i = (1 - (i-center)/(N/2))^p
    
    def make_power(p):
        i = np.arange(N)
        center = (N - 1) / 2.0
        # Normalize distance to [-1, 1]
        dist = np.abs(i - center) / (N / 2.0)
        # Ensure positive
        h = np.power(np.maximum(1 - dist, 0.0001), p)
        return h

    res_pow = minimize_scalar(lambda p: _score(make_power(p)), bounds=(0.1, 10), method='bounded')
    if res_pow.success:
        h_opt = make_power(res_pow.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # 2. Local Search on Real Values
    # Start from the best parametric window found.
    # Use Nelder-Mead or a simple coordinate descent.
    # N=400 is too high for full Nelder-Mead in 150s if steps are expensive.
    # But we can use a gradient-free method with a limited number of dimensions?
    # Or just a few iterations of random perturbations.
    
    h_cur = best_h.copy()
    val_cur = best_val
    
    # Time check
    if time.time() > deadline - 10:
        return h_cur
    
    # Coordinate Descent with Random Steps
    rng = random.Random(42)
    step_size = 0.01
    iterations = 2000 # Rough estimate, check time
    
    for _ in range(iterations):
        if time.time() > deadline - 5:
            break
        
        i = rng.randint(0, N - 1)
        old_h = h_cur[i]
        # Try a few perturbations
        new_h_val = old_h + rng.uniform(-step_size, step_size)
        h_cur[i] = new_h_val
        
        new_val = _score(h_cur)
        
        if new_val < val_cur:
            val_cur = new_val
            best_h = h_cur.copy()
            # Shrink step size slightly on success
            step_size *= 0.95
        else:
            h_cur[i] = old_h
            # Increase step size slightly on failure (exploration)
            step_size = min(step_size * 1.05, 0.5)

    return best_h

# EVOLVE-BLOCK-END
