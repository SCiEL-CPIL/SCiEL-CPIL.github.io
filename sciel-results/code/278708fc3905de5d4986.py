import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
DEADLINE = 110.0  # safety margin before 150s kill

# Grid setup
dx_step = 0.5 / N  # 0.01
step_centers = -0.25 + (np.arange(N) + 0.5) * dx_step

# Fine grid for continuous step-function autocorrelation
M = 4096
fine_dx = 0.5 / M
fine_x = -0.25 + (np.arange(M) + 0.5) * fine_dx
fine_to_step = np.clip(np.floor((fine_x + 0.25) / dx_step).astype(int), 0, N - 1)

# Precompute FFT size
_fft_n = 1
while _fft_n < 2 * M:
    _fft_n *= 2


def _autocorr_max(f_fine):
    """Compute max of autoconvolution using FFT."""
    g = np.zeros(_fft_n)
    g[:M] = f_fine
    F = np.fft.rfft(g)
    ac = np.fft.irfft(F * F.conj(), n=_fft_n)
    return float(np.max(ac))


def _f_from_h(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    int_f2 = dx_step * float(np.sum(h * h))
    int_f = dx_step * float(np.sum(h))
    if int_f <= 0:
        return 0.0
    f_fine = h[fine_to_step]
    acmax = _autocorr_max(f_fine)
    if acmax <= 0:
        return 0.0
    return int_f2 / (int_f * acmax)


def _objective(x):
    """Negative ratio for minimization."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx  # peak = 1
    return -_f_from_h(h)


def _param_shapes():
    """Generate diverse parameterized shape initializations."""
    shapes = []
    
    # Gaussian
    for sig in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        shapes.append(np.exp(-step_centers**2 / (2 * sig * sig)))
    
    # Lorentzian
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        shapes.append(1.0 / (1.0 + (step_centers / sig)**2))
    
    # Triangle
    for w in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / w, 0, None)
        if np.max(s) > 0:
            shapes.append(s)
    
    # Plateau / box
    for w in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        s = (np.abs(step_centers) < w).astype(float)
        shapes.append(s)
    
    # Single step (center)
    s = np.zeros(N); s[N // 2] = 1.0; shapes.append(s)
    
    # Two steps symmetric
    for d in [1, 2, 3, 5, 7, 10]:
        s = np.zeros(N)
        if N // 2 - d >= 0 and N // 2 + d < N:
            s[N // 2 - d] = 1
            s[N // 2 + d] = 1
        shapes.append(s)
    
    # Three steps symmetric
    for d in [2, 3, 5]:
        s = np.zeros(N)
        if N // 2 - d >= 0 and N // 2 + d < N:
            s[N // 2 - d] = 1
            s[N // 2] = 1
            s[N // 2 + d] = 1
        shapes.append(s)
    
    # Polynomial-based shapes
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        t = 2 * step_centers  # map to [-0.5, 0.5]
        c = np.polynomial.chebyshev.chebval(t, [0] * deg + [1])
        c = np.maximum(c, 0)
        if np.max(c) > 0:
            shapes.append(c / np.max(c))
    
    # Sine-based
    for freq in [1, 2, 3, 4, 5, 6]:
        s = np.sin(freq * np.pi * step_centers)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    # Cosine-based (centered)
    for freq in [1, 2, 3, 4]:
        s = np.cos(freq * np.pi * step_centers)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    # Smoothstep-like
    for w in [0.05, 0.1, 0.15, 0.2]:
        t = np.clip((w - np.abs(step_centers)) / (2 * w), 0, 1)
        s = t * t * (3 - 2 * t)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    # Power-law decay
    for p in [1.5, 2, 2.5, 3, 4, 5]:
        s = np.exp(-p * np.abs(step_centers))
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    # Bump functions (compact support, smooth)
    for w in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / w
        s = np.where(t < 1, np.exp(-1 / (1 - t**2)), 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    # Random smooth shapes
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Random smooth via low-freq Fourier
        freqs = rng.integers(1, 8, size=8)
        amps = rng.random(8)
        phases = rng.random(8) * 2 * np.pi
        s = np.zeros(N)
        for f, a, ph in zip(freqs, amps, phases):
            s += a * np.cos(f * 2 * np.pi * step_centers + ph)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))
    
    return shapes


def _polish(h, t0, deadline=DEADLINE):
    """Final high-precision polish pass."""
    if time.time() - t0 > deadline - 5:
        return h, _f_from_h(h)
    try:
        res = minimize(_objective, h, method='L-BFGS-B',
                       bounds=[(0.0, 10.0)] * N,
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200})
        x = np.maximum(res.x, 0.0)
        mx = np.max(x)
        if mx <= 0:
            return h, _f_from_h(h)
        h2 = x / mx
        f2 = _f_from_h(h2)
        f1 = _f_from_h(h)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass
    return h, _f_from_h(h)


def _coordinate_descent(h, t0, deadline=DEADLINE, max_iter=50):
    """Simple coordinate descent to refine the solution."""
    h = np.array(h, dtype=float)
    best_f = _f_from_h(h)
    
    for it in range(max_iter):
        if time.time() - t0 > deadline - 10:
            break
        improved = False
        for i in range(N):
            # Try perturbing coordinate i
            orig = h[i]
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                h[i] = max(0.0, orig + delta)
                f = _f_from_h(h)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break
    
    return h, best_f


def construct():
    best_h = np.ones(N)
    best_f = _f_from_h(best_h)
    t0 = time.time()
    
    starts = _param_shapes()
    
    # Add more random starts
    rng = np.random.default_rng(0)
    for _ in range(15):
        starts.append(rng.random(N))
    
    # Add starts based on combinations of features
    for _ in range(10):
        base = rng.random(N)
        # Smooth it
        kernel = np.exp(-np.arange(-5, 6)**2 / 5)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode='same')
        base = np.maximum(base, 0)
        if np.max(base) > 0:
            starts.append(base / np.max(base))
    
    bounds = [(0.0, 10.0)] * N
    
    # First pass: standard optimization on all starts
    for idx, s in enumerate(starts):
        if time.time() - t0 > DEADLINE - 30:
            break
        try:
            s = np.array(s, dtype=float)
            mx = np.max(s)
            if mx <= 0:
                continue
            s = s / mx  # normalize
            
            res = minimize(_objective, s, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            x = np.maximum(res.x, 0.0)
            mx = np.max(x)
            if mx <= 0:
                continue
            h = x / mx
            f = _f_from_h(h)
            if f > best_f:
                best_f = f
                best_h = h.copy()
        except Exception:
            pass
    
    # Second pass: coordinate descent on best
    if time.time() - t0 < DEADLINE - 15:
        best_h, best_f = _coordinate_descent(best_h, t0)
    
    # Third pass: polish with tighter tolerances
    if time.time() - t0 < DEADLINE - 5:
        best_h, best_f = _polish(best_h, t0)
    
    # Final: try a few more random restarts from perturbed best if time permits
    if time.time() - t0 < DEADLINE - 10:
        rng2 = np.random.default_rng(123)
        for _ in range(5):
            if time.time() - t0 > DEADLINE - 10:
                break
            h_pert = best_h + rng2.normal(0, 0.1, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            try:
                res = minimize(_objective, h_pert, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                x = np.maximum(res.x, 0.0)
                mx = np.max(x)
                if mx <= 0:
                    continue
                h2 = x / mx
                f2 = _f_from_h(h2)
                if f2 > best_f:
                    best_f = f2
                    best_h = h2.copy()
            except Exception:
                pass
    
    # Ensure nonneg
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
