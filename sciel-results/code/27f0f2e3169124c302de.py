"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf)
over 50 nonnegative step heights using spectral parameterization h = |FFT(z)|^2
with L-BFGS-B optimization and smooth-max for the supremum norm."""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # safety margin before 120s kill


def _time_left():
    return _DEADLINE - time.time()


def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)


def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0.0)
    l2sq = np.sum(h ** 2)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _smooth_max(x, sigma=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max <= 0:
        return 0.0
    x_shifted = (x - x_max) / sigma
    # Clip to avoid overflow in exp
    x_shifted = np.clip(x_shifted, -100, 0)
    log_sum_exp = x_max + sigma * np.log(np.sum(np.exp(x_shifted)))
    return log_sum_exp


def _objective(z_flat):
    """
    Objective to minimize: negative of the ratio.
    z_flat is a real vector of length 2*N representing complex z.
    h = |FFT(z)|^2, normalized to L1 = 1.
    """
    re = z_flat[:N]
    im = z_flat[N:]
    z = re + 1j * im
    fft_z = np.fft.fft(z)
    h = np.abs(fft_z) ** 2
    
    # Normalize to L1 = 1 for scale invariance and stability
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.01)
    
    if peak < 1e-30:
        return 0.0
    
    return -l2sq / peak


def _z_to_heights(z):
    """Convert complex vector z to nonnegative heights via |FFT(z)|^2, normalized to L1=1."""
    z = np.asarray(z, dtype=np.complex128)
    fft_z = np.fft.fft(z)
    h = np.abs(fft_z) ** 2
    l1 = np.sum(h)
    if l1 > 0:
        h = h / l1
    return h


def _heights_to_z(h):
    """Convert nonnegative heights to a complex vector z such that |FFT(z)|^2 ~ h.
    Uses zero phases: z = IFFT(sqrt(h))."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    mags = np.sqrt(h)
    z = np.fft.ifft(mags)
    return z


def _local_optimize_z(z0, max_iter=500):
    """Optimize the complex vector z using L-BFGS-B."""
    x0 = np.concatenate([z0.real, z0.imag])
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 
                                'gtol': 1e-10, 'maxfun': 3000})
        z_opt = res.x[:N] + 1j * res.x[N:]
        h_opt = _z_to_heights(z_opt)
        score = _ratio(h_opt)
        return h_opt, score
    except Exception:
        h0 = _z_to_heights(z0)
        return h0, _ratio(h0)


def _initializations():
    """Generate diverse initial candidates."""
    candidates = []
    x = np.linspace(-1, 1, N)
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Gaussian profiles with various widths
    for width in [0.1, 0.3, 0.5, 1.0, 2.0]:
        h = np.exp(-x**2 / width)
        candidates.append(h)
    
    # 3. Bimodal
    candidates.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    candidates.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    
    # 4. Triangle / tent
    candidates.append(1 - np.abs(x))
    
    # 5. Cosine window
    candidates.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Random smooth profiles
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        candidates.append(h_smooth)
    
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    candidates.append(h_center)
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    candidates.append(h_edge)
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    candidates.append(h_alt)
    
    # 10. Linear ramp
    candidates.append(np.linspace(0.1, 1.0, N))
    candidates.append(np.linspace(1.0, 0.1, N))
    
    # 11. DFT-based random
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft)
        candidates.append(h_dft)
    
    # 12. More structured patterns
    candidates.append(np.sin(np.pi * (x + 1) / 2)**2)
    candidates.append(np.cos(np.pi * x / 2)**2)
    
    # 13. Sparse-like patterns
    for sparsity in [0.1, 0.3, 0.5]:
        h = np.zeros(N)
        n_nonzero = int(sparsity * N)
        positions = np.random.RandomState(99).choice(N, n_nonzero, replace=False)
        h[positions] = 1.0
        candidates.append(h)
    
    return candidates


def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Evaluate initial candidates and find best
    candidates = _initializations()
    for h in candidates:
        if _time_left() < 5:
            break
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Convert best to spectral representation and optimize
    z_best = _heights_to_z(best_h)
    
    # Optimize from the best initial candidate
    if _time_left() > 10:
        h_opt, s_opt = _local_optimize_z(z_best, max_iter=500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            z_best = _heights_to_z(best_h)
    
    # Try optimization from several diverse starting points
    # Sort candidates by their raw score and try the top ones
    scored_candidates = []
    for h in candidates:
        s = _ratio(h)
        scored_candidates.append((s, h))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Try optimization from top diverse candidates
    tried = set()
    for s_cand, h_cand in scored_candidates:
        if _time_left() < 8:
            break
        # Check if this candidate is "close" to already tried ones
        is_new = True
        for _, h_tried in tried:
            if np.sum((h_cand - h_tried)**2) < 0.5:
                is_new = False
                break
        if not is_new:
            continue
        
        z_cand = _heights_to_z(h_cand)
        # Add small random perturbation
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        z_pert = z_cand + (rng.randn(N) + 1j * rng.randn(N)) * 0.05
        
        h_opt, s_opt = _local_optimize_z(z_pert, max_iter=400)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            z_best = _heights_to_z(best_h)
        
        tried.append((s_cand, h_cand))
    
    # Final polish: perturb around the best solution
    if _time_left() > 8:
        for pert_seed in [100, 200, 300, 400, 500]:
            if _time_left() < 6:
                break
            rng_local = np.random.RandomState(pert_seed)
            z_pert = z_best + (rng_local.randn(N) + 1j * rng_local.randn(N)) * 0.1
            
            h_opt, s_opt = _local_optimize_z(z_pert, max_iter=300)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
                z_best = _heights_to_z(best_h)
    
    # Ensure nonnegative and return
    best_h = np.maximum(0, best_h)
    
    return list(best_h)


if __name__ == "__main__":
    result = construct()
    print("Score:", _ratio(result))
    print("Heights:", result)
