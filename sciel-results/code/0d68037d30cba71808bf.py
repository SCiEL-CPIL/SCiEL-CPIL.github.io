"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Structural change: use the DFT magnitude-squared parameterization h_k = |P(omega^k)|^2.
This representation naturally enforces non-negativity and exploits the spectral
structure of the autoconvolution: the autoconvolution's peak is controlled by the
autocorrelation of the spectrum, which can be flattened by optimizing the phases.

We optimize the real and imaginary parts of the complex spectrum P, using the
log-sum-exp smoothed supremum for differentiability, with L-BFGS-B.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (1/2)/DH

_DEADLINE = 145.0
_SAFETY = 8.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _h_from_spectrum(params):
    """Reconstruct h from complex spectrum parameters (real, imag interleaved or separate).
    params is a flat array of length 2*N: first N are real parts, next N are imag parts.
    h_k = |FFT^{-1}(P)|^2, but we use the DFT parameterization:
    Actually, the standard approach: h is the inverse DFT of |P|^2? No.
    
    The DFT parameterization h_k = |P(omega^k)|^2 means h is the pointwise square of
    the magnitude of the DFT of some signal? Or is h the signal whose DFT has magnitude P?
    
    Let's use the formulation from inspiration 2:
    h = |g|^2 where g is a complex vector of length N.
    But the key insight is that h is nonnegative by construction.
    We optimize g (real and imaginary parts).
    """
    real_parts = params[:N]
    imag_parts = params[N:]
    g = real_parts + 1j * imag_parts
    h = np.abs(g) ** 2
    return h


def _soft_logsup(h, beta):
    """Smooth log(sup autoconv) via log-sum-exp."""
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 1e-30:
        return -np.inf
    shifted = ac - acmax
    return acmax + math.log(float(np.sum(np.exp(beta * shifted)))) / beta


def _neg_logR_spectrum(params, beta):
    """Negative log ratio for spectrum parameterization.
    We fix ||h||_1 = 1 by normalizing, so we maximize ||h||_2^2 / ||h*h||_inf.
    log R = log(l2sq) - log(l1) - log(sup). With l1=1, log R = log(l2sq) - log(sup).
    But we want to minimize -log R.
    """
    h = _h_from_spectrum(params)
    l1 = float(np.sum(h))
    if l1 <= 1e-30:
        return 1e12
    # Normalize to l1 = 1
    h = h / l1
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-30:
        return 1e12
    ls = _soft_logsup(h, beta)
    if not math.isfinite(ls):
        return 1e12
    # log R = log(l2sq) - log(1) - ls = log(l2sq) - ls
    return -(math.log(l2sq) - ls)


def _neg_logR_spectrum_grad(params, beta):
    """Numerical gradient for spectrum parameterization."""
    val = _neg_logR_spectrum(params, beta)
    g = np.zeros_like(params)
    eps = 1e-6
    for i in range(len(params)):
        pp = params.copy()
        pp[i] += eps
        vp = _neg_logR_spectrum(pp, beta)
        pm = params.copy()
        pm[i] -= eps
        vm = _neg_logR_spectrum(pm, beta)
        g[i] = (vp - vm) / (2 * eps)
    return val, g


def _initializations_spectrum():
    """Yield diverse starting points for the spectrum parameters."""
    inits = []
    rng = np.random.default_rng(42)

    # Uniform magnitude, random phases
    for _ in range(20):
        real = rng.randn(N) * 0.1
        imag = rng.randn(N) * 0.1
        inits.append(np.concatenate([real, imag]))

    # Structured: low-frequency dominant
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        k = np.arange(N)
        mag = np.exp(-alpha * k / N)
        real = mag * np.cos(rng.uniform(0, 2 * np.pi, N))
        imag = mag * np.sin(rng.uniform(0, 2 * np.pi, N))
        inits.append(np.concatenate([real, imag]))

    # High-frequency dominant
    for alpha in [0.5, 1.0, 2.0]:
        k = np.arange(N)
        mag = np.exp(-alpha * (N - k) / N)
        real = mag * np.cos(rng.uniform(0, 2 * np.pi, N))
        imag = mag * np.sin(rng.uniform(0, 2 * np.pi, N))
        inits.append(np.concatenate([real, imag]))

    # Bandpass
    for center in [10, 20, 30]:
        for width in [5, 10, 15]:
            k = np.arange(N)
            mag = np.exp(-0.5 * ((k - center) / width) ** 2)
            real = mag * np.cos(rng.uniform(0, 2 * np.pi, N))
            imag = mag * np.sin(rng.uniform(0, 2 * np.pi, N))
            inits.append(np.concatenate([real, imag]))

    # From good h initializations: compute g such that |g|^2 ~ h
    # Sidon-like: sparse h
    sidon_idx = [0, 1, 4, 9, 16, 25, 36, 49]
    h_sidon = np.zeros(N)
    for i in sidon_idx:
        h_sidon[i] = 1.0
    h_sidon = h_sidon / np.sum(h_sidon)
    g_sidon = np.sqrt(h_sidon)
    inits.append(np.concatenate([g_sidon, np.zeros(N)]))

    # Two spikes
    for a in [5, 15, 25]:
        for b in [a + 15, a + 20, a + 25]:
            if b < N:
                h2 = np.zeros(N)
                h2[a] = 1.0
                h2[b] = 1.0
                h2 = h2 / np.sum(h2)
                g2 = np.sqrt(h2)
                inits.append(np.concatenate([g2, np.zeros(N)]))

    # Random smooth h -> g
    for seed in range(15):
        rng_local = np.random.default_rng(seed)
        h_rand = rng_local.random(N)
        h_rand = np.convolve(h_rand, np.ones(5) / 5, mode='same')
        h_rand = h_rand / np.sum(h_rand)
        g_rand = np.sqrt(h_rand)
        phase = rng_local.uniform(0, 2 * np.pi, N)
        real = g_rand * np.cos(phase)
        imag = g_rand * np.sin(phase)
        inits.append(np.concatenate([real, imag]))

    return inits


def _local_optimize_spectrum(params0, time_budget):
    """Optimize the spectrum parameters. Returns (best_params, best_ratio)."""
    t0 = time.time()
    best_params = params0.copy()
    h_best = _h_from_spectrum(params0)
    best_v = _ratio(h_best)

    def _over():
        return (time.time() - t0) > time_budget

    beta = 40.0
    try:
        res = minimize(
            lambda p: _neg_logR_spectrum(p, beta),
            params0,
            method='L-BFGS-B',
            jac=lambda p: _neg_logR_spectrum_grad(p, beta),
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda p: (np.nan if _over() else p),
        )
        if res.x is not None:
            h_res = _h_from_spectrum(res.x)
            v_res = _ratio(h_res)
            if v_res > best_v:
                best_v = v_res
                best_params = res.x.copy()
    except Exception:
        pass

    # Polish with Nelder-Mead on the raw ratio
    try:
        for _ in range(2):
            if _over() or _time_left() <= 5.0:
                break
            res2 = minimize(
                lambda p: -_ratio(_h_from_spectrum(p)),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True, 'disp': False},
            )
            h_res2 = _h_from_spectrum(res2.x)
            v_res2 = _ratio(h_res2)
            if v_res2 > best_v:
                best_v = v_res2
                best_params = res2.x.copy()
    except Exception:
        pass

    return best_params, best_v


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    inits = _initializations_spectrum()
    scored = []
    for p in inits:
        h = _h_from_spectrum(p)
        v = _ratio(h)
        scored.append((v, p))
        if v > best_v:
            best_v = v
            best_h = h.copy()
    scored.sort(key=lambda t: -t[0])

    budget_per = 5.0
    for v0, p in scored:
        if _time_left() <= budget_per + 2.0:
            break
        if v0 < best_v * 0.5 and best_v > 5.0:
            continue
        p_opt, v_opt = _local_optimize_spectrum(p, time_budget=budget_per)
        h_opt = _h_from_spectrum(p_opt)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
        if _time_left() <= 20.0:
            break

    # Final polish
    if best_h is not None and _time_left() > 15.0:
        # Convert best_h back to spectrum params for polish
        g = np.sqrt(np.maximum(best_h, 0.0))
        p_best = np.concatenate([g, np.zeros(N)])
        try:
            p_pol, v_pol = _local_optimize_spectrum(p_best, time_budget=_time_left() - 18.0)
            h_pol = _h_from_spectrum(p_pol)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
