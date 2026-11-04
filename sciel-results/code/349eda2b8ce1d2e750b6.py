"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = (1.0 / 2.0) / (N - 1)  # spacing = 1/98

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * sup|h*h|) for nonnegative step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    
    l2_sq = np.sum(h ** 2) * DELTA
    l1 = np.sum(h) * DELTA
    
    if l1 < 1e-30:
        return 0.0
    
    # Autoconvolution of step function: (h*h)(t) = integral h(x)h(t-x) dx
    # For step functions on equal bins, compute via correlation
    # The supremum for nonnegative h is at t=0
    # (h*h)(0) = integral h(x)^2 dx = DELTA * sum(h^2)
    # But wait - for step functions, the convolution at 0 involves overlap of bins
    # Let's compute properly: h(x) = h_i for x in [x_i - d/2, x_i + d/2]
    # (h*h)(t) = sum_i sum_j h_i h_j * overlap_length(bin_i, bin_j shifted by t)
    
    # At t=0: overlap of bin_i with itself = d, so (h*h)(0) = d * sum(h_i^2)
    # But also cross terms? No - at t=0, bin_i overlaps with bin_j only if i=j for equal spacing
    # Actually for adjacent bins there's partial overlap... let me think again.
    
    # For equal-width bins at positions x_i = -1/4 + i*DELTA, width DELTA:
    # bin_i = [x_i - DELTA/2, x_i + DELTA/2]
    # At t=0, bin_i overlaps bin_j by max(0, DELTA - |x_i - x_j|)
    # Since |x_i - x_j| = |i-j|*DELTA, overlap = max(0, DELTA - |i-j|*DELTA)
    # = DELTA * (1 - |i-j|) for |i-j| < 1, i.e., only i=j gives DELTA
    # Wait: |i-j|*DELTA < DELTA means |i-j| < 1, so only i=j.
    # So (h*h)(0) = DELTA * sum(h_i^2).
    
    # But the supremum might not be at 0! For nonnegative functions it IS at 0
    # because (h*h)(t) <= (h*h)(0) by rearrangement? Not necessarily for step functions.
    # Actually for nonnegative h, (h*h)(0) = integral h^2, and for t != 0,
    # (h*h)(t) = integral h(x)h(x+t) dx which by Cauchy-Schwarz <= sqrt(integral h^2 * integral h^2) = (h*h)(0)
    # So yes, sup = (h*h)(0) = DELTA * sum(h^2) for nonnegative h.
    
    sup_conv = DELTA * np.sum(h ** 2)
    
    if sup_conv < 1e-30:
        return 0.0
    
    return l2_sq / (l1 * sup_conv)


def _negative_ratio(h):
    return -_compute_ratio(h)


def _dft_initialization(k, phase_shift=0.0):
    """Generate a nonnegative initialization using DFT structure."""
    n = N
    freqs = np.fft.rfftfreq(n)
    # Create a spectrum with energy concentrated at low frequencies
    spectrum = np.zeros(n // 2 + 1, dtype=np.complex128)
    spectrum[0] = 1.0
    for j in range(1, min(k, n // 2 + 1)):
        spectrum[j] = np.exp(1j * phase_shift * j) * (1.0 / (1.0 + j * 0.5))
    
    # Reconstruct signal
    signal = np.fft.irfft(spectrum * n, n=n)
    # Make nonnegative
    signal = np.abs(signal)
    if np.max(signal) > 0:
        signal = signal / np.max(signal)
    return signal


def _gaussian_initialization(center, width):
    """Generate a Gaussian-like nonnegative initialization."""
    x = np.linspace(-0.25, 0.25, N)
    h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h / np.max(h) if np.max(h) > 0 else np.ones(N) / N


def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    best_h = np.ones(N)
    best_val = _compute_ratio(best_h)
    
    # Define constraints: h >= 0
    bounds = [(0.0, None)] * N
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Gaussian at various positions and widths
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.05, 0.1, 0.15, 0.2]:
            initializations.append(_gaussian_initialization(center, width))
    
    # DFT-based initializations
    for k in [2, 3, 4, 5, 6, 8, 10]:
        for phase in [0.0, 0.5, 1.0, 1.5]:
            initializations.append(_dft_initialization(k, phase))
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        h_rand = rng.random(N)
        initializations.append(h_rand)
    
    # Random sparse
    for _ in range(10):
        h_sparse = np.zeros(N)
        indices = rng.choice(N, size=rng.integers(5, 15), replace=False)
        h_sparse[indices] = rng.random(len(indices))
        initializations.append(h_sparse)
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
        
        init = np.clip(init, 0, None)
        if np.all(init == 0):
            continue
        
        try:
            result = minimize(
                _negative_ratio,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_val:
                h_opt = np.clip(result.x, 0, None)
                val = _compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Second pass: refine best solution with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _negative_ratio,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.clip(result.x, 0, None)
            val = _compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Third pass: try local perturbations around best
    if time.time() < deadline - 15.0:
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            h_pert = best_h + rng.normal(0, 0.1, N)
            h_pert = np.clip(h_pert, 0, None)
            try:
                result = minimize(
                    _negative_ratio,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                h_opt = np.clip(result.x, 0, None)
                val = _compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    return list(best_h)
