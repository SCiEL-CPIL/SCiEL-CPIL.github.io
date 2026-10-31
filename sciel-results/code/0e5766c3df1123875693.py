import numpy as np
import time

N = 600

def _autoconv_peak(f):
    """Compute the maximum of the autoconvolution of f (discrete, full)."""
    # Use FFT for efficiency
    n = len(f)
    size = 2 * n - 1
    # Zero-pad to size for linear convolution
    f_padded = np.zeros(size)
    f_padded[:n] = f
    F = np.fft.rfft(f_padded, n=size)
    ac = np.fft.irfft(F * F, n=size)
    return np.max(ac)

def _normalize(f):
    """Normalize: max(autoconv) / (sum(f))^2 * N"""
    s = np.sum(f)
    if s == 0:
        return np.inf
    peak = _autoconv_peak(f)
    return peak / (s * s) * N

def _make_cosine_window(alpha=1.0):
    """Cosine-tapered window: f[i] = (0.5 + 0.5*cos(2*pi*alpha*i/(N-1)))^p"""
    i = np.arange(N)
    t = i / (N - 1)
    f = (0.5 + 0.5 * np.cos(2 * np.pi * alpha * t))
    return np.maximum(f, 0)

def _make_gaussian_window(sigma):
    """Gaussian window centered."""
    i = np.arange(N)
    center = (N - 1) / 2.0
    f = np.exp(-0.5 * ((i - center) / sigma) ** 2)
    return f

def _make_flat_top(width_frac):
    """Flat top with cosine edges."""
    i = np.arange(N)
    f = np.ones(N)
    edge = int(N * width_frac / 2)
    if edge > 0:
        for j in range(edge):
            t = j / edge
            f[j] = 0.5 + 0.5 * np.cos(np.pi * t)
            f[N - 1 - j] = 0.5 + 0.5 * np.cos(np.pi * t)
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_f = np.ones(N)
    best_score = _normalize(best_f)
    
    # Try various parameterized families
    candidates = []
    
    # Cosine windows with different alpha
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        candidates.append(_make_cosine_window(alpha))
    
    # Gaussian windows with different sigmas
    for sigma in [N*0.1, N*0.15, N*0.2, N*0.25, N*0.3, N*0.4, N*0.5]:
        candidates.append(_make_gaussian_window(sigma))
    
    # Flat top with different edge widths
    for wf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        candidates.append(_make_flat_top(wf))
    
    # Power-law windows
    for p in [1.0, 1.5, 2.0, 2.5, 3.0]:
        i = np.arange(N)
        t = i / (N - 1) * 2 - 1  # -1 to 1
        f = np.maximum(1.0 - np.abs(t) ** p, 0)
        candidates.append(f)
    
    # Evaluate candidates
    for f in candidates:
        if time.time() > deadline:
            break
        if np.all(f < 0):
            continue
        f = np.maximum(f, 0)
        score = _normalize(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Local refinement: perturb the best candidate
    if time.time() < deadline - 20:
        # Try small perturbations
        rng = np.random.RandomState(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            f_pert = best_f.copy()
            # Add small random noise
            noise = rng.normal(0, 0.05, N)
            f_pert = f_pert + noise
            f_pert = np.maximum(f_pert, 0)
            score = _normalize(f_pert)
            if score < best_score:
                best_score = score
                best_f = f_pert.copy()
    
    # Convert to list
    return [float(x) for x in best_f]
