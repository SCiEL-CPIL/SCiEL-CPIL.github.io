import numpy as np
import time

N = 400
DEADLINE = 120.0


def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes (excluding main lobe)."""
    f = np.asarray(f, dtype=float)
    n = f.size
    # Use FFT for efficient autoconvolution with zero-padding
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    F = np.fft.rfft(f, m)
    ac = np.fft.irfft(F * F.conj(), m)[:2 * n - 1]
    # ac is the full linear autocorrelation, centered at index n-1
    center = n - 1
    # Exclude the main lobe (a small window around center)
    # Main lobe width depends on signal; exclude center ± 2
    main_lobe_width = 2
    ac_sidelobes = ac.copy()
    ac_sidelobes[center - main_lobe_width:center + main_lobe_width + 1] = 0.0
    return float(np.max(np.abs(ac_sidelobes)))


def _grad_max_sidelobe(f):
    """Subgradient of max_sidelobe |ac[k]| w.r.t. f."""
    f = np.asarray(f, dtype=float)
    n = f.size
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    F = np.fft.rfft(f, m)
    ac = np.fft.irfft(F * F.conj(), m)[:2 * n - 1]
    center = n - 1
    main_lobe_width = 2
    ac_masked = ac.copy()
    ac_masked[center - main_lobe_width:center + main_lobe_width + 1] = 0.0
    k = int(np.argmax(np.abs(ac_masked)))
    sign = np.sign(ac_masked[k]) if ac_masked[k] != 0 else 1.0
    # Gradient of sign * ac[k] w.r.t. f
    # ac[k] = sum_i f[i] * f[i+k] (for k>=0, with appropriate indexing)
    g = np.zeros(n)
    if k == 0:
        g[:] = 2.0 * sign * f
    else:
        # ac[k] = sum_{i=0}^{n-k-1} f[i+k] * f[i]
        # d/d f[j] = sign * (f[j-k] + f[j+k]) where indices are valid
        g[:n - k] += sign * f[k:]
        g[k:] += sign * f[:n - k]
    return g


def _kaiser_seed(n, beta=10.0):
    """Generate a Kaiser-Bessel window as initial seed."""
    from scipy.signal import windows
    w = windows.kaiser(n, beta)
    # Center it and make it zero-mean
    w = w - w.mean()
    return w


def _polish(x0, max_iter=3000, tol=1e-10):
    """Gradient descent on the max sidelobe objective."""
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_val = _get_max_sidelobe(x)
    g = _grad_max_sidelobe(x)
    g_norm = float(np.linalg.norm(g, 2)) + 1e-9
    lr = 1.0 / (g_norm + 1e-12)
    it = 0
    t0 = time.time()
    while it < max_iter:
        if time.time() > DEADLINE - 5.0:
            break
        x = x - lr * g
        # Normalize to keep scale reasonable
        x_norm = np.linalg.norm(x)
        if x_norm > 1e-9:
            x = x / x_norm * np.sqrt(N)
        f = _get_max_sidelobe(x)
        if f < best_val - tol:
            best_val = f
            best_x = x.copy()
        g = _grad_max_sidelobe(x)
        g_norm = float(np.linalg.norm(g, 2)) + 1e-9
        lr = 1.0 / (g_norm + 1e-12)
        it += 1
    return best_x, best_val


def construct():
    t0 = time.time()
    # Try multiple seeds with different beta values
    best_x = None
    best_val = float('inf')
    
    for beta in [8.0, 10.0, 12.0, 15.0]:
        if time.time() > DEADLINE - 10.0:
            break
        base = _kaiser_seed(N, beta)
        x, val = _polish(base, max_iter=1500)
        if val < best_val:
            best_val = val
            best_x = x
    
    # If no good result yet, try a random smooth signal
    if best_x is None or time.time() < DEADLINE - 20.0:
        rng = np.random.default_rng(42)
        for trial in range(3):
            if time.time() > DEADLINE - 10.0:
                break
            # Random smooth signal: sum of low-frequency sines
            freqs = rng.integers(1, 10, size=5)
            phases = rng.uniform(0, 2 * np.pi, size=5)
            amps = rng.uniform(0.5, 1.0, size=5)
            t = np.linspace(0, 2 * np.pi, N)
            base = np.zeros(N)
            for i in range(5):
                base += amps[i] * np.sin(freqs[i] * t + phases[i])
            base = base - base.mean()
            x, val = _polish(base, max_iter=1000)
            if val < best_val:
                best_val = val
                best_x = x
    
    if best_x is None:
        return _kaiser_seed(N, 10.0)
    
    return best_x
