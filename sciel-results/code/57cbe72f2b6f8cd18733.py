"""Combined program: spectral parameterization with global stochastic search and local refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 400

def _compute_value(h):
    """Compute the exact task value."""
    conv = np.convolve(h, h)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconv_fft(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c

def _hard_value(h):
    """Fast hard value using FFT."""
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    # === Phase 1: Analytical pattern search (fast, diverse initializations) ===
    i = np.arange(N)
    patterns = []

    # Basic patterns
    patterns.append(np.ones(N))
    patterns.append(np.linspace(1, 2, N))
    patterns.append(1.0 + np.linspace(0, 1, N) ** 2)

    # Low-frequency cosine modes with DC offset
    for k in [1, 2, 3, 4, 5, 8, 10]:
        for a in [0.1, 0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7]:
            h = 1.0 + a * np.cos(2.0 * np.pi * k * i / N)
            patterns.append(h)

    # Sine modes
    for k in [1, 2, 3, 5]:
        for a in [0.3, 0.5, 0.7]:
            h = 1.0 + a * np.sin(2.0 * np.pi * k * i / N)
            patterns.append(h)

    # Two-level with gap (negative middle)
    for gap_frac in [0.1, 0.2, 0.3, 0.4]:
        gap_len = int(N * gap_frac)
        h = np.ones(N)
        h[N // 2 - gap_len // 2:N // 2 + gap_len // 2] = -0.5
        patterns.append(h)

    # Block patterns
    for block_frac in [0.2, 0.3, 0.5]:
        block_len = int(N * block_frac)
        h = np.zeros(N)
        h[:block_len] = 1.0
        patterns.append(h)

    # Spectral parameterization: h[i] = A + sum_j B_j * cos(2*pi*k_j*i/N + phi_j)
    # Try a few fixed spectral configs
    for A in [0.5, 1.0, 1.5]:
        for k1 in [1, 2, 3]:
            for B1 in [0.1, 0.3, 0.5]:
                h = A + B1 * np.cos(2 * np.pi * k1 * i / N)
                patterns.append(h)
                h = A + B1 * np.cos(2 * np.pi * k1 * i / N) + 0.2 * np.cos(4 * np.pi * k1 * i / N)
                patterns.append(h)

    # Evaluate all patterns
    for h in patterns:
        if not np.all(np.isfinite(h)):
            continue
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    if time.time() > deadline:
        return best_h

    # === Phase 2: Local optimization with reduced-dimensional parameterization ===
    # Parameterize h as a0 + a1*cos(2*pi*i/N) + b1*sin(2*pi*i/N) +
    #                  a2*cos(4*pi*i/N) + b2*sin(4*pi*i/N) +
    #                  a3*cos(6*pi*i/N) + b3*sin(6*pi*i/N)
    # This gives a smooth, low-frequency structure that suppresses high-frequency peaks.

    def spectral_h(params):
        a0 = params[0]
        a1, b1 = params[1], params[2]
        a2, b2 = params[3], params[4]
        a3, b3 = params[5], params[6]
        h = (a0 +
             a1 * np.cos(2 * np.pi * i / N) + b1 * np.sin(2 * np.pi * i / N) +
             a2 * np.cos(4 * np.pi * i / N) + b2 * np.sin(4 * np.pi * i / N) +
             a3 * np.cos(6 * np.pi * i / N) + b3 * np.sin(6 * np.pi * i / N))
        return h

    def spectral_objective(params):
        if time.time() > deadline:
            return best_val
        h = spectral_h(params)
        if not np.all(np.isfinite(h)):
            return 1e10
        return _hard_value(h)

    # Fit best_h to spectral parameterization
    A_mat = np.column_stack([
        np.ones(N),
        np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N),
        np.cos(4 * np.pi * i / N), np.sin(4 * np.pi * i / N),
        np.cos(6 * np.pi * i / N), np.sin(6 * np.pi * i / N)
    ])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, best_h, rcond=None)
        x0 = coeffs
    except:
        x0 = np.array([1.0, 0.1, 0.0, 0.05, 0.0, 0.02, 0.0])

    # Multiple restarts for local optimization
    for restart in range(5):
        if time.time() > deadline - 20:
            break

        if restart == 0:
            x_init = x0
        else:
            # Perturb the best solution found so far
            x_init = x0 + 0.1 * np.random.randn(7)

        try:
            res = minimize(
                spectral_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                h_candidate = spectral_h(res.x)
                val_candidate = _compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
                    x0 = res.x.copy()
        except Exception:
            continue

    if time.time() > deadline:
        return best_h

    # === Phase 3: Global stochastic search in spectral space ===
    # Use random restarts with Nelder-Mead for global exploration

    def global_objective(params):
        if time.time() > deadline:
            return best_val
        h = spectral_h(params)
        if not np.all(np.isfinite(h)):
            return 1e10
        return _hard_value(h)

    rng = np.random.RandomState(42)
    for attempt in range(20):
        if time.time() > deadline - 30:
            break

        # Random initialization in spectral space
        x_rand = np.array([
            rng.uniform(0.5, 2.0),       # DC offset
            rng.uniform(-0.5, 0.5),       # cos mode 1
            rng.uniform(-0.5, 0.5),       # sin mode 1
            rng.uniform(-0.3, 0.3),       # cos mode 2
            rng.uniform(-0.3, 0.3),       # sin mode 2
            rng.uniform(-0.2, 0.2),       # cos mode 3
            rng.uniform(-0.2, 0.2),       # sin mode 3
        ])

        try:
            res = minimize(
                global_objective,
                x_rand,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 3000}
            )
            if res.fun < best_val:
                h_candidate = spectral_h(res.x)
                val_candidate = _compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
        except Exception:
            continue

    if time.time() > deadline:
        return best_h

    # === Phase 4: Direct local optimization on full h vector (polish) ===
    # Use L-BFGS-B on the full 400-dim vector from the best spectral solution

    def full_objective(h):
        if time.time() > deadline:
            return best_val
        if not np.all(np.isfinite(h)):
            return 1e10
        return _hard_value(h)

    try:
        res = minimize(
            full_objective,
            best_h.copy(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.fun < best_val:
            val_candidate = _compute_value(res.x)
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = res.x.copy()
    except Exception:
        pass

    # Final validation
    final_val = _compute_value(best_h)
    if final_val < best_val:
        best_val = final_val

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)

    return best_h
