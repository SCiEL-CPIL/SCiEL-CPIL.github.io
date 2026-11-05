"""
Strategy: Zadoff-Chu based representation with smooth windowing for the third autocorrelation inequality.

We represent the step heights as the real part of a Zadoff-Chu sequence (a CAZAC sequence)
multiplied by a smooth low-frequency window function. Zadoff-Chu sequences have ideal
periodic autocorrelation (zero off-peak), but finite-length truncation creates sidelobes.
By applying a smooth window (Hann or Gaussian), we suppress these sidelobes while
preserving the near-zero autocorrelation in the central region.

The key insight is that Zadoff-Chu sequences have a special structure where the
autocorrelation sidelobes are bounded, and windowing can further reduce them.

Parameters to optimize:
- root_index: which Zadoff-Chu root to use (integer, affects the sequence pattern)
- window_width: width of the Gaussian window (controls sidelobe suppression)
- center_offset: shift of the window center
- amplitude: overall scaling
- offset: DC offset to keep sum away from zero
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = 140.0  # wall-clock limit with safety margin


def _zadoff_chu_sequence(root, length):
    """
    Generate a Zadoff-Chu sequence of given length.
    h[n] = exp(-j * pi * root * n * (n + 1) / length) for n = 0, ..., length-1
    """
    n = np.arange(length, dtype=float)
    # For even length, use (n+1) to avoid the singularity at n=0 for some roots
    # The standard definition: z[n] = exp(-j*pi*root*n*(n+1)/L) for L even
    phase = -np.pi * root * n * (n + 1) / length
    return np.exp(1j * phase)


def _zch_based_h(params):
    """
    Generate step heights from Zadoff-Chu + window parameters.
    params = [root_index, window_width, center_offset, amplitude, offset]
    
    root_index: continuous parameter, we'll round to nearest valid root
    window_width: std of Gaussian window (larger = less windowing)
    center_offset: shift of window center (in samples)
    amplitude: scaling of the ZC sequence
    offset: DC offset
    """
    root_index, window_width, center_offset, amplitude, offset = params
    
    # Clamp root to a valid range (roots 1 to N//2 - 1 are typically used)
    root = int(round(np.clip(root_index, 1, N // 2 - 1)))
    
    # Generate Zadoff-Chu sequence
    zc = _zadoff_chu_sequence(root, N)
    
    # Create Gaussian window
    n = np.arange(N, dtype=float)
    center = N / 2.0 + center_offset
    # window_width is the standard deviation; larger means less aggressive windowing
    # Clip to avoid numerical issues
    ww = max(abs(window_width), 1.0)
    window = np.exp(-0.5 * ((n - center) / ww) ** 2)
    
    # Combine: real part of (amplitude * ZC * window) + offset
    h = amplitude * np.real(zc * window) + offset
    
    return h


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _objective(params):
    """Objective function for optimization."""
    h = _zch_based_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Best solution so far
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0

    # Define bounds for differential evolution
    bounds = [
        (1.0, N // 2 - 1),    # root_index
        (10.0, 200.0),        # window_width (std of Gaussian)
        (-N / 4, N / 4),      # center_offset
        (-3.0, 3.0),          # amplitude
        (-1.5, 1.5),          # offset
    ]

    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-5,
                polish=False,
                seed=42,
                init='sobol',
            )
            if de_result.success or _objective(de_result.x) < best_val:
                h_de = _zch_based_h(de_result.x)
                val_de = _evaluate(h_de)
                if np.isfinite(val_de) and val_de < best_val:
                    best_val = val_de
                    best_h = h_de
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            # Get the best params from DE
            x0 = de_result.x.copy()
            # Ensure root_index is rounded for the local search
            x0[0] = round(x0[0])
            
            nm_result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
            )
            if np.all(np.isfinite(nm_result.x)):
                h_nm = _zch_based_h(nm_result.x)
                val_nm = _evaluate(h_nm)
                if np.isfinite(val_nm) and val_nm < best_val:
                    best_val = val_nm
                    best_h = h_nm
        except Exception:
            pass

    # Phase 3: Multi-start with different initial parameters
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            root = rng.integers(1, N // 2 - 1)
            ww = rng.uniform(15.0, 150.0)
            center = rng.uniform(-N / 4, N / 4)
            amp = rng.uniform(-2.5, 2.5)
            off = rng.uniform(-1.0, 1.0)
            x0 = np.array([float(root), ww, center, amp, off])
            try:
                nm = minimize(
                    _objective, x0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7},
                )
                if np.all(np.isfinite(nm.x)):
                    h_c = _zch_based_h(nm.x)
                    v_c = _evaluate(h_c)
                    if np.isfinite(v_c) and v_c < best_val:
                        best_val = v_c
                        best_h = h_c
            except Exception:
                pass

    # Phase 4: Final greedy local search on the best h
    if time.time() < deadline - 3:
        best_val = _evaluate(best_h)
        rng = np.random.default_rng(999)
        for _ in range(50000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = best_h.copy()
            h_new[idx] += rng.normal(0, 0.02)
            v_new = _evaluate(h_new)
            if np.isfinite(v_new) and v_new < best_val - 1e-12:
                best_val = v_new
                best_h = h_new

    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)

    return best_h
