"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Strategy:
1. Use a narrow rectangular pulse centered at the origin as the primary signal shape,
   since the ratio scales as ~1/w for narrow pulses.
2. Optimize over the pulse width and center position, plus a few side lobe parameters.
3. Use direct L-BFGS-B optimization on h with non-negativity bounds.
4. Also try DFT-based parameterization for spectral structure.
5. Sort initializations by score and optimize from the best ones.
6. Final high-precision polish pass.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)


def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15:
        return 0.0
    l2sq = np.sum(h ** 2)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _ratio_neg(h):
    """Negative of the ratio for minimization."""
    return -_ratio(h)


def _narrow_rect_pulse(center, width, N=50, grid_start=-0.25, grid_end=0.25):
    """Create a narrow rectangular pulse."""
    x = np.linspace(grid_start, grid_end, N, endpoint=False) + (grid_end - grid_start) / N
    h = np.zeros(N)
    mask = np.abs(x - center) < width / 2.0
    h[mask] = 1.0
    return h


def _gaussian_profile(center, width, N=50, grid_start=-0.25, grid_end=0.25):
    """Generate a Gaussian profile on the grid."""
    x = np.linspace(grid_start, grid_end, N, endpoint=False) + (grid_end - grid_start) / N
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _lorentzian_profile(center, gamma, N=50, grid_start=-0.25, grid_end=0.25):
    """Generate a Lorentzian (Cauchy) profile on the grid."""
    x = np.linspace(grid_start, grid_end, N, endpoint=False) + (grid_end - grid_start) / N
    return gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _sinc_like_profile(width, center=0.0, N=50, grid_start=-0.25, grid_end=0.25):
    """Create a sinc-like profile."""
    x = np.linspace(grid_start, grid_end, N, endpoint=False) + (grid_end - grid_start) / N
    t = (x - center) / width
    return np.maximum(np.sinc(t) ** 2, 0)


def _two_gaussian(c1, w1, c2, w2, h1=1.0, h2=1.0, N=50, grid_start=-0.25, grid_end=0.25):
    """Sum of two Gaussians."""
    x = np.linspace(grid_start, grid_end, N, endpoint=False) + (grid_end - grid_start) / N
    h1_arr = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    h2_arr = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    return h1_arr + h2_arr


def _local_optimize(h0, time_limit=15.0):
    """Direct optimization of h with non-negativity constraint using L-BFGS-B."""
    h = np.maximum(h0, 0.0).copy()
    start = time.time()
    
    def objective(h_flat):
        if time.time() - start > time_limit:
            return _ratio_neg(h_flat)
        return _ratio_neg(h_flat)
    
    try:
        result = minimize(objective, h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000})
        h_opt = np.maximum(result.x, 0.0)
    except Exception:
        h_opt = h
    return h_opt


def _dft_parameterize(z):
    """Parameterize heights as |DFT(z)|^2 where z is a complex vector."""
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h


def _dft_inverse(h):
    """Given h = |DFT(z)|^2, find a z (phase choice)."""
    H_mags = np.sqrt(h + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z


def _optimize_from_z(z0, time_limit=10.0):
    """Optimize over complex z to maximize ratio via |DFT(z)|^2."""
    z = z0.astype(complex).copy()
    start = time.time()
    
    def objective(z_flat):
        if time.time() - start > time_limit:
            z_c = z_flat[:N] + 1j * z_flat[N:]
            h = _dft_parameterize(z_c)
            return _ratio_neg(h)
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        return _ratio_neg(h)
    
    z_flat = np.concatenate([z.real, z.imag])
    
    try:
        result = minimize(objective, z_flat, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000})
        z_opt = result.x[:N] + 1j * result.x[N:]
        h_opt = _dft_parameterize(z_opt)
    except Exception:
        h_opt = _dft_parameterize(z)
    return h_opt


def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations, prioritizing narrow pulses
    initializations = []
    
    # Narrow rectangular pulses at various widths and centers
    for width in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            initializations.append(_narrow_rect_pulse(center, width))
    
    # Narrow Gaussians
    for width in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.15, -0.1, 0.0, 0.1, 0.15]:
            initializations.append(_gaussian_profile(center, width))
    
    # Lorentzians
    for gamma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.1, 0.0, 0.1]:
            initializations.append(_lorentzian_profile(center, gamma))
    
    # Sinc-like profiles
    for width in [0.02, 0.05, 0.1, 0.15]:
        for center in [0.0, 0.05, -0.05]:
            initializations.append(_sinc_like_profile(width, center))
    
    # Two Gaussians (bimodal)
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.05, 0.05)]:
        for w in [0.02, 0.03, 0.05, 0.08]:
            initializations.append(_two_gaussian(c1, w, c2, w, 1.0, 1.0))
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Random smooth
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(3) / 3
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations.append(h_smooth)
    
    # DFT-based random
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H) ** 2
        h_dft = h_dft / (np.max(h_dft) + 1e-15)
        initializations.append(h_dft)
    
    # Sparse single-peak initializations
    for pos in [N//4, N//2, 3*N//4, N//5, 4*N//5]:
        h_sparse = np.zeros(N)
        h_sparse[pos] = 1.0
        initializations.append(h_sparse)
    
    # Sort initializations by score, keep top ones for optimization
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        if np.sum(h_init) < 1e-15:
            continue
        s = _ratio(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best from initializations
    if scored_inits:
        if scored_inits[0][0] > best_score:
            best_score = scored_inits[0][0]
            best_h = scored_inits[0][1].copy()
    
    # Optimize from top initializations
    top_k = min(20, len(scored_inits))
    for i in range(top_k):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 20:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization
        h_opt = _local_optimize(h_init, time_limit=8.0)
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # DFT-based optimization
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:
            break
        
        z_init = _dft_inverse(h_init)
        h_dft_opt = _optimize_from_z(z_init, time_limit=6.0)
        score = _ratio(h_dft_opt)
        if score > best_score:
            best_score = score
            best_h = h_dft_opt.copy()
    
    # Final polish: high-precision optimization from best found
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        h_polish = _local_optimize(best_h, time_limit=15.0)
        score = _ratio(h_polish)
        if score > best_score:
            best_score = score
            best_h = h_polish.copy()
        
        # Another DFT polish
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 5:
            z_polish = _dft_inverse(best_h)
            h_dft_polish = _optimize_from_z(z_polish, time_limit=8.0)
            score = _ratio(h_dft_polish)
            if score > best_score:
                best_score = score
                best_h = h_dft_polish.copy()
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
