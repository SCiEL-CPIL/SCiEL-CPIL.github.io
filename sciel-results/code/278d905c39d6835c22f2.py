import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -1.0
    h = np.clip(h, 0, None)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    # autoconvolution via FFT for speed
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac_full = np.fft.irfft(H * H.conj(), n=n_fft)
    sup_ac = np.max(np.abs(ac_full))
    if sup_ac < 1e-15:
        return -1.0
    return l2_sq / (l1 * sup_ac)


def _neg_ratio(h):
    r = _ratio(h)
    if r < 0:
        return 1.0
    return -r


def construct():
    global _start_time
    _start_time = time.time()

    best_h = np.ones(N)
    best_val = _ratio(best_h)

    # Generate diverse initial guesses focusing on sparse/concentrated shapes
    _single_spike = np.zeros(N)
    _single_spike[N // 2] = 1.0
    _two_spikes = np.zeros(N)
    _two_spikes[N // 2 - 1] = 1.0
    _two_spikes[N // 2] = 1.0
    _three_spikes = np.zeros(N)
    _three_spikes[N // 2 - 1] = 1.0
    _three_spikes[N // 2] = 1.0
    _three_spikes[N // 2 + 1] = 1.0

    # Geometric decay from center
    _geo_center = np.zeros(N)
    center = N // 2
    for i in range(N):
        _geo_center[i] = 0.7 ** abs(i - center)

    # Power law decay
    _power_law = np.zeros(N)
    for i in range(N):
        _power_law[i] = 1.0 / (1.0 + 0.5 * abs(i - center))

    # Asymmetric geometric
    _asym_geo = np.zeros(N)
    for i in range(N):
        if i <= center:
            _asym_geo[i] = 0.6 ** (center - i)
        else:
            _asym_geo[i] = 0.3 ** (i - center)

    initial_guesses = [
        _single_spike,
        _two_spikes,
        _three_spikes,
        _geo_center,
        _power_law,
        _asym_geo,
        np.ones(N),
        np.exp(-0.5 * (np.arange(N) - N // 2) ** 2 / (N // 6) ** 2),
        np.ones(N) * (1 + 0.5 * np.cos(2 * np.pi * np.arange(N) / N)),
        np.array([1.0] * 10 + [0.1] * 30 + [1.0] * 10),
        np.array([0.0] * 5 + [1.0] * 40 + [0.0] * 5),
    ]

    for guess in initial_guesses:
        if _elapsed() > _DEADLINE - 20:
            break
        guess = np.clip(guess, 0, None)
        if np.sum(guess) < 1e-10:
            continue
        # Normalize to avoid scale issues (ratio is scale-invariant)
        guess = guess / np.sum(guess)

        # Use Nelder-Mead for robustness
        res = minimize(_neg_ratio, guess, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12,
                                'adaptive': True})
        if _elapsed() > _DEADLINE - 5:
            break
        h_opt = np.clip(res.x, 0, None)
        val = _ratio(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt

    # Try more random restarts if time permits
    rng = np.random.default_rng(42)
    for _ in range(40):
        if _elapsed() > _DEADLINE - 5:
            break
        # Random log-normal-ish shapes with emphasis on sparse structures
        base = rng.random(N)
        exponent = rng.uniform(0.5, 5.0)
        base = base ** exponent
        # Sometimes zero out some entries to encourage sparsity
        if rng.random() < 0.3:
            mask = rng.random(N) < 0.3
            base[mask] = 0.0
        base = np.clip(base, 0, None)
        if np.sum(base) < 1e-10:
            continue
        base = base / np.sum(base)
        res = minimize(_neg_ratio, base, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12,
                                'adaptive': True})
        h_opt = np.clip(res.x, 0, None)
        val = _ratio(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt

    # Final local refinement with COBYLA on the best solution
    if _elapsed() < _DEADLINE - 10:
        res = minimize(_neg_ratio, best_h, method='COBYLA',
                       options={'maxiter': 2000, 'rhobeg': 1e-4, 'catol': 1e-14})
        h_opt = np.clip(res.x, 0, None)
        val = _ratio(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt

    return list(best_h)
