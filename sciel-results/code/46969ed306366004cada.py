"""Combined program for the Third Autocorrelation Inequality task.

Strategy:
1. Spectral (CAZAC-like) and parametric initializations.
2. Scale-invariance exploitation: fix sum(h)=1, minimize peak of autoconvolution.
3. Projected subgradient / Frank-Wolfe style descent on the peak constraint.
4. Nelder-Mead + L-BFGS-B refinement with soft-max surrogate.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    ac = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(ac) / (s * s))


def _conv(h):
    return np.convolve(h, h, mode='full')


def _normalize_sum1(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h + 1.0  # fallback
    return h / s


def _peak_subgradient_step(h, lr):
    """One projected subgradient step minimizing max(conv(h,h)) with sum=1.
    Returns updated h with sum 1."""
    conv = _conv(h)
    m = np.max(conv)
    # gradient of max w.r.t. h: for the argmax lag k, d max/dh[i] = h[i] + h[i-k] (with bounds)
    k = int(np.argmax(conv))
    # full conv index k corresponds to lag k - (N-1)
    lag = k - (N - 1)
    g = np.zeros(N)
    if lag >= 0:
        g[:N - lag] += h[lag:]
        g[lag:] += h[:N - lag]
    else:
        g[-lag:] += h[:N + lag]
        g[:N + lag] += h[-lag:]
    # gradient of the squared peak (m * g) to make step scale-aware
    h_new = h - lr * m * g
    # project back to sum=1
    h_new = _normalize_sum1(h_new)
    return h_new, m


def _projected_descent(h0, max_iter, lr0, time_budget):
    """Projected subgradient descent minimizing peak(autoconv) subject to sum=1."""
    h = _normalize_sum1(h0.copy())
    t_start = time.time()
    best_h = h.copy()
    best_peak = np.max(_conv(h))
    lr = lr0
    stuck = 0
    for it in range(max_iter):
        if time.time() - t_start > time_budget:
            break
        h_new, m = _peak_subgradient_step(h, lr)
        # accept if peak decreased, else reduce lr
        if m < best_peak - 1e-12:
            best_peak = m
            best_h = h_new.copy()
            h = h_new
            stuck = 0
        else:
            h = h_new
            lr *= 0.7
            stuck += 1
            if stuck > 30:
                # restart with small perturbation
                h = _normalize_sum1(h + 0.01 * np.random.randn(N))
                lr = lr0
                stuck = 0
    return best_h, best_peak


def _soft_max(x, beta):
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) * beta))) / beta


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill
    rng = np.random.default_rng(12345)
    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        if abs(np.sum(h)) < 1e-10:
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # ---------- Phase 1: Spectral / CAZAC-like initialization ----------
    # CAZAC: constant amplitude, zero autocorrelation (cyclic). We approximate
    # by designing the spectrum |H|^2 to be flat (which gives flat cyclic
    # autocorrelation), then take real part.
    for trial in range(30):
        if time.time() > deadline - 80:
            break
        # random spectral mask with flat magnitude target
        nfreq = N // 2 + 1
        # random phases, flat magnitude
        phases = 2 * np.pi * rng.random(nfreq)
        H = np.exp(1j * phases)
        # force Hermitian symmetry for real output
        Hf = np.zeros(N, dtype=complex)
        Hf[:nfreq] = H
        Hf[N - (nfreq - 1):] = np.conj(H[1:nfreq][::-1])
        h = np.fft.ifft(Hf).real
        # add DC to ensure nonzero sum
        h = h + 0.5
        try_candidate(h)

    # Gaussian spectral profiles (from inspiration 1)
    for trial in range(20):
        if time.time() > deadline - 80:
            break
        freqs = np.fft.rfftfreq(N)
        bandwidth = rng.uniform(0.05, 0.4)
        center = rng.uniform(-0.3, 0.3)
        H = np.exp(-((freqs - center) ** 2) / (2 * bandwidth ** 2))
        H *= (1.0 + 0.3 * rng.randn(len(H)))
        h = np.fft.irfft(H, n=N)
        try_candidate(h)

    # ---------- Phase 2: Parametric candidates ----------
    candidates = []
    # Cosine / sine modes
    for k in range(1, 25):
        for t in np.linspace(-1.2, 1.2, 13):
            candidates.append(1.0 + t * np.cos(2.0 * np.pi * k * idx / N))
            candidates.append(1.0 + t * np.sin(2.0 * np.pi * k * idx / N))
    # Chirp
    for mu in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for t in np.linspace(-1.0, 1.0, 7):
            phase = 2.0 * np.pi * (idx / N + mu * (idx / N) ** 2)
            candidates.append(1.0 + t * np.cos(phase))
    # Two-frequency mix
    for k1 in [1, 2, 3, 5, 7]:
        for k2 in [2, 3, 4, 6, 8, 10]:
            for t1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for t2 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    candidates.append(
                        1.0 + t1 * np.cos(2.0 * np.pi * k1 * idx / N)
                        + t2 * np.sin(2.0 * np.pi * k2 * idx / N))
    # Quadratic / ramp
    x = idx / (N - 1)
    for t in np.linspace(-0.9, 0.9, 13):
        candidates.append(1.0 + t * (2.0 * x - 1.0) ** 2)
        candidates.append(1.0 + t * x)

    for h in candidates:
        if time.time() > deadline - 60:
            break
        try_candidate(h)

    # ---------- Phase 3: Projected subgradient descent from top seeds ----------
    # Collect a few good starting points
    start_points = [best_h.copy()]
    # add some perturbed versions
    for _ in range(6):
        if time.time() > deadline - 50:
            break
        pert = best_h.copy() + 0.1 * rng.randn(N)
        start_points.append(pert)
    # add a couple of distinct parametric seeds
    start_points.append(1.0 + 0.8 * np.cos(2.0 * np.pi * 3 * idx / N))
    start_points.append(1.0 + 0.8 * np.sin(2.0 * np.pi * 5 * idx / N))

    for sp in start_points:
        if time.time() > deadline - 20:
            break
        try:
            h_ref, peak = _projected_descent(sp, max_iter=2000, lr0=0.01,
                                             time_budget=12.0)
            try_candidate(h_ref)
        except Exception:
            pass

    # ---------- Phase 4: Nelder-Mead + L-BFGS-B refinement with soft-max ----------
    def make_obj(beta):
        def obj(x):
            conv = _conv(x)
            sm = _soft_max(conv, beta)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty
        return obj

    h0 = _normalize_sum1(best_h.copy())
    for beta in [150.0, 300.0, 500.0]:
        if time.time() > deadline - 10:
            break
        try:
            obj = make_obj(beta)
            res = minimize(obj, h0, method='L-BFGS-B',
                           options={'maxiter': 150, 'maxfun': 500, 'gtol': 1e-7})
            if time.time() > deadline - 5:
                break
            try_candidate(res.x)
            h0 = _normalize_sum1(res.x.copy())
        except Exception:
            pass

    # Nelder-Mead from best
    if time.time() < deadline - 8:
        try:
            obj = make_obj(300.0)
            res = minimize(obj, h0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-8})
            try_candidate(res.x)
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
