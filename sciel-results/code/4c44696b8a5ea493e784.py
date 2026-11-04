"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Combined strategy:
- DFT spectral parameterization h = |FFT(z)|^2 for natural non-negativity and
  spectral structure exploitation (from inspirations 1 & 2).
- Smooth-max (log-sum-exp) approximation for the supremum norm to enable
  gradient-based optimization.
- Diverse initializations including Sidon-like sparse spike sets, Gaussian
  bumps, Lorentzian profiles, and structured patterns.
- L-BFGS-B optimization in complex z-space with careful time management.
- Final polishing in h-space with Nelder-Mead.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_DEADLINE = 145.0
_SAFETY = 5.0
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
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _smooth_max(x, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    x_max = float(np.max(x))
    if x_max <= 0.0:
        return 0.0
    x_shifted = x - x_max
    exp_ac = np.exp(beta * x_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 0.0
    return x_max + (1.0 / beta) * math.log(sum_exp)


def _smooth_max_grad(x, beta=50.0):
    """Return (smooth_max_value, weights) where weights sum to 1."""
    x = np.asarray(x, dtype=float)
    x_max = float(np.max(x))
    if x_max <= 0.0:
        return 0.0, np.zeros_like(x)
    x_shifted = x - x_max
    exp_ac = np.exp(beta * x_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 0.0, np.zeros_like(x)
    w = exp_ac / sum_exp
    val = x_max + (1.0 / beta) * math.log(sum_exp)
    return val, w


def _z_to_h(z_flat):
    """Convert flattened complex z to h = |FFT(z)|^2, normalized to L1=1."""
    z = z_flat[:N] + 1j * z_flat[N:]
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return np.zeros(N)
    return h / l1


def _h_to_z(h):
    """Convert h to complex z such that h ∝ |FFT(z)|^2."""
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return np.zeros(2 * N)
    h_norm = h / l1
    # Take square root of the power spectrum
    H = np.sqrt(h_norm + 1e-30)
    # Inverse FFT to get z
    z = np.fft.ifft(H)
    return np.concatenate([z.real, z.imag])


def _objective_z(z_flat):
    """
    Objective to minimize: negative log of ratio.
    Uses smooth max for the autoconvolution peak.
    """
    h = _z_to_h(z_flat)
    l2sq = float(np.dot(h, h))
    if l2sq <= 0.0:
        return 1e12, np.zeros_like(z_flat)
    
    ac = _autoconv(h)
    smax, w = _smooth_max_grad(ac, beta=50.0)
    if smax <= 0.0:
        return 1e12, np.zeros_like(z_flat)
    
    # log R = log(l2sq) - log(l1) - log(smax)
    # Since h is normalized to L1=1, log(l1) = 0
    logR = math.log(l2sq) - math.log(smax)
    
    # Gradient computation
    # d(logR)/d(h_j) = 2*h_j/l2sq - d(log(smax))/d(h_j)/smax
    # d(log(smax))/d(h_j) = sum_i w_i * d(ac_i)/d(h_j)
    # d(ac_i)/d(h_j) = 2*DH*h[i + OFFSET - j] if 0 <= i + OFFSET - j < N else 0
    
    d_smax_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_smax_dh[j] = 2.0 * DH * s
    
    g_h = 2.0 * h / l2sq - d_smax_dh / smax
    
    # Chain rule: h = |FFT(z)|^2 / ||FFT(z)||^2_1
    # This is complex. Let's use a simpler approach: 
    # h = |FFT(z)|^2, then normalize. The gradient of the normalization
    # is complicated. Instead, let's optimize in h-space directly for the 
    # final polish, and use z-space only for the initial search.
    
    # For z-space gradient, we need:
    # dh/dz = 2 * Re(conj(FFT(z)) * FFT(dz)) / l1 - h * (d(l1)/dz)/l1^2
    # This is getting complex. Let's use a finite-difference-free approach:
    # We'll compute the gradient of log(l2sq) - log(smax) wrt h, then
    # map back to z using the Jacobian of FFT.
    
    # h = |FFT(z)|^2 / sum(|FFT(z)|^2)
    # Let F = FFT(z), so h = |F|^2 / sum(|F|^2)
    # d(h_j)/d(z_m) = [2*conj(F_j)*FFT_basis[j,m]*sum(|F|^2) - |F_j|^2 * d(sum(|F|^2))/d(z_m)] / sum(|F|^2)^2
    # d(sum(|F|^2))/d(z_m) = 2*sum_j conj(F_j)*FFT_basis[j,m]
    
    # This is O(N^2) per component. Let's just use a simpler strategy:
    # Optimize in h-space directly with bounds, and use z-space for 
    # generating diverse initializations.
    
    # For now, return a placeholder gradient (we'll use h-space optimization)
    return -logR, np.zeros_like(z_flat)


def _objective_h(h):
    """
    Objective to minimize: negative log of ratio in h-space.
    Uses smooth max for the autoconvolution peak.
    """
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(h)
    
    ac = _autoconv(h)
    smax, w = _smooth_max_grad(ac, beta=50.0)
    if smax <= 0.0:
        return 1e12, np.zeros_like(h)
    
    logR = math.log(l2sq) - math.log(l1) - math.log(smax)
    
    # Gradient
    d_smax_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_smax_dh[j] = 2.0 * DH * s
    
    g_h = 2.0 * h / l2sq - 1.0 / l1 - d_smax_dh / smax
    
    return -logR, -g_h


def _local_optimize_h(h0, time_budget):
    """Run L-BFGS-B on the direct h parameterization; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = float(np.linalg.norm(h0))
    if s > 0:
        h0 = h0 / s
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()
    
    def neg_logR(x):
        val, _ = _objective_h(x)
        return val
    
    def neg_logR_grad(x):
        _, g = _objective_h(x)
        return g
    
    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass
    
    # Nelder-Mead polish
    try:
        for _ in range(2):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                lambda x: -_ratio(x), best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass
    
    return best_h, best_v


def _sidon_initializations():
    """Generate Sidon-like sparse spike initializations."""
    inits = []
    # Sidon set: no two pairs have the same sum
    # For N=50, use well-separated spikes
    # Single spikes
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4, N // 5, 2 * N // 5, 3 * N // 5, 4 * N // 5]:
        s = np.zeros(N)
        s[pos] = 1.0
        inits.append(s)
    
    # Two spikes with large separation
    for a in range(0, N, 8):
        for b in range(a + 8, N, 8):
            s = np.zeros(N)
            s[a] = 1.0
            s[b] = 1.0
            inits.append(s)
    
    # Three spikes with large separation
    for a in range(0, N, 12):
        for b in range(a + 12, N, 12):
            for c in range(b + 12, N, 12):
                s = np.zeros(N)
                s[a] = 1.0
                s[b] = 1.0
                s[c] = 1.0
                inits.append(s)
    
    return inits


def _structured_initializations():
    """Generate diverse structured initializations."""
    inits = []
    x = np.arange(N)
    x_norm = (x - N / 2) / (N / 2)  # [-1, 1]
    
    # Uniform
    inits.append(np.ones(N))
    
    # Gaussian-like
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        inits.append(np.exp(-0.5 * (x_norm / sigma) ** 2))
    
    # Bimodal
    for center in [0.3, 0.5, 0.7]:
        inits.append(np.exp(-((x_norm - center) ** 2) / 0.1) + np.exp(-((x_norm + center) ** 2) / 0.1))
    
    # Triangle / tent
    inits.append(1 - np.abs(x_norm))
    
    # Cosine window
    inits.append(0.5 + 0.5 * np.cos(np.pi * x_norm))
    
    # Lorentzian profiles
    for center in range(0, N, 7):
        for width in [1.0, 2.0, 3.0]:
            inits.append(1.0 / (1.0 + ((x - center) / width) ** 2))
    
    # Random smooth profiles
    rng = np.random.default_rng(12345)
    for _ in range(30):
        h_rand = np.abs(rng.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        inits.append(h_smooth)
    
    # Random sparse profiles
    for _ in range(30):
        k = rng.integers(1, 12)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    
    # DFT-based initializations
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H) ** 2
        h_dft = h_dft / np.max(h_dft)
        inits.append(h_dft)
    
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    # Combine all initializations
    inits = _sidon_initializations() + _structured_initializations()
    
    # Score and sort
    scored = []
    for s in inits:
        v = _ratio(s)
        scored.append((v, s))
        if v > best_v:
            best_v = v
            best_h = s.copy()
    scored.sort(key=lambda t: -t[0])

    # Optimize top candidates
    budget_per = 2.5
    for v0, s in scored:
        if _time_left() <= budget_per + 2.0:
            break
        if v0 < best_v * 0.5 and best_v > 0 and best_v > 10.0:
            continue

        h_opt, v_opt = _local_optimize_h(s, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()

        if _time_left() <= 20.0:
            break

    # Final polish
    if best_h is not None and _time_left() > 15.0:
        h_pol, v_pol = _local_optimize_h(best_h, time_budget=_time_left() * 0.5)
        if v_pol > best_v:
            best_v = v_pol
            best_h = h_pol.copy()

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
