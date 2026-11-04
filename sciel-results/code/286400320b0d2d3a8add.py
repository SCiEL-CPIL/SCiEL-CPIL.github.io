"""Minimize normalized max autoconvolution of a nonnegative step function on 600 intervals."""
import time
import math
import numpy as np

N = 600
H = 1.0 / N          # interval width
START = time.time()
DEADLINE = 120.0
MARGIN = 12.0        # safety margin (s)
NLAG = 2 * N - 1     # 1199 convolution bins


def _time_left():
    return (START + DEADLINE - MARGIN) - time.time()


def _fft_conv(f):
    """Full linear autoconvolution c[k] for k=0..2N-2 (c[k] = sum_i f[i]*f[i+k])."""
    fs = np.fft.rfft(f, n=2 * N)
    return np.fft.irfft(fs * fs, n=2 * N)[:NLAG]


def _score(f):
    c = _fft_conv(f)
    c0 = c[0]
    if c0 <= 0:
        return float("inf")
    return float(np.max(c) / c0)


def _softmax(c, c0, beta):
    """Smooth surrogate of max(c)/c0 using log-sum-exp."""
    m = np.max(c)
    return (m + np.log(np.sum(np.exp(beta * (c - m)))) / beta) / c0


def _softmax_grad(c, c0, beta):
    """d/dc of soft-max(c)/c0 (c0 held fixed)."""
    m = np.max(c)
    ex = np.exp(beta * (c - m))
    p = ex / np.sum(ex)
    return p / c0


# ---------------- Phase 1: Gaussian superposition (smooth low-dim search) ----------------
def _gauss_f(psi, centers, sigmas, scale):
    """f[i] = scale * sum_j psi_j * exp(-0.5*((x_i - c_j)/s_j)^2), psi_j >= 0."""
    x = np.arange(N) * H
    out = np.zeros(N)
    for j in range(len(centers)):
        out += psi[j] * np.exp(-0.5 * ((x - centers[j]) / sigmas[j]) ** 2)
    return scale * out


def _phase1():
    rng = np.random.default_rng(1234)
    ncomp = 4
    # initial centers spread across domain, sigmas a few intervals wide
    centers = np.array([-0.12, -0.04, 0.05, 0.13])
    sigmas = np.array([0.03, 0.04, 0.035, 0.03])
    psi = np.ones(ncomp) * 0.5
    scale = 1.0

    def f_of(v):
        psi = np.exp(v[:ncomp])          # enforce psi > 0
        ctr = np.tanh(v[ncomp:ncomp + ncomp]) * 0.25
        sig = 0.01 + 0.05 * np.exp(v[2 * ncomp:3 * ncomp])
        sc = 1.0 + 0.5 * np.exp(v[3 * ncomp])
        return _gauss_f(psi, ctr, sig, sc), psi, ctr, sig, sc

    # gradient descent on log-sum-exp surrogate
    v = np.zeros(4 * ncomp + 1)
    v[:ncomp] = np.log(0.5)
    v[ncomp:ncomp + ncomp] = np.arctanh(centers / 0.25)
    v[2 * ncomp:3 * ncomp] = np.log((sigmas - 0.01) / 0.05)
    v[3 * ncomp] = np.log(0.5)

    beta = 60.0
    lr = 0.02
    best_f = f_of(v)[0]
    best_s = _score(best_f)

    for it in range(4000):
        if _time_left() < 8.0:
            break
        f, psi, ctr, sig, sc = f_of(v)
        c = _fft_conv(f)
        c0 = c[0]
        obj = _softmax(c, c0, beta)
        if obj < best_s:
            best_s = obj
            best_f = f.copy()

        # numerical gradient (cheap enough: 4*ncomp+1 ~ 17 evals per iter)
        g = np.zeros_like(v)
        hstep = 1e-4
        for k in range(len(v)):
            vp = v.copy(); vp[k] += hstep
            fm = v.copy(); fm[k] -= hstep
            cp = _fft_conv(f_of(vp)[0]); cm = _fft_conv(f_of(fm)[0])
            op = _softmax(cp, cp[0], beta)
            om = _softmax(cm, cm[0], beta)
            g[k] = (op - om) / (2 * hstep)
        v = np.clip(v - lr * g, -20.0, 20.0)
        lr *= 0.999
    return best_f, best_s


# ---------------- Phase 2: direct bin-level polish (projected gradient + active set) ----------------
def _phase2(f0, s0):
    f = f0.copy()
    best_f = f.copy()
    best_s = s0
    beta = 40.0
    lr = 1e-3
    for it in range(6000):
        if _time_left() < 6.0:
            break
        c = _fft_conv(f)
        c0 = c[0]
        obj = _softmax(c, c0, beta)
        if obj < best_s:
            best_s = obj
            best_f = f.copy()

        # gradient of soft-max(c)/c0 w.r.t. f
        p = _softmax_weights(c, beta)          # normalized weights
        grad_c = p / c0
        # c[k] = sum_i f[i] f[i+k]; dc/df[i] = sum_k grad_c[k] * (f[i-k] + f[i+k])
        g = np.zeros(N)
        for k in range(NLAG):
            gc = grad_c[k]
            if gc == 0:
                continue
            # indices i where i+k in [0,N-1] and i in [0,N-1]
            i0 = max(0, -k)
            i1 = min(N, N - k)
            if i1 <= i0:
                continue
            g[i0:i1] += gc * f[i0 + k:i1 + k]          # f[i+k]
            g[i0:i1] += gc * f[i0 - k:i1 - k]          # f[i-k]
        # projected gradient step with nonnegativity
        f = np.maximum(f - lr * g, 0.0)
        lr *= 0.9995
    return best_f, best_s


def _softmax_weights(c, beta):
    m = np.max(c)
    ex = np.exp(beta * (c - m))
    return ex / np.sum(ex)


# ---------------- Phase 3: high-precision polish (exact local solve on active bins) ----------------
def _phase3(f0, s0):
    f = f0.copy()
    best_f = f.copy()
    best_s = s0
    beta = 200.0
    lr = 1e-4
    for it in range(4000):
        if _time_left() < 3.0:
            break
        c = _fft_conv(f)
        c0 = c[0]
        obj = _softmax(c, c0, beta)
        if obj < best_s:
            best_s = obj
            best_f = f.copy()
        p = _softmax_weights(c, beta)
        grad_c = p / c0
        g = np.zeros(N)
        for k in range(NLAG):
            gc = grad_c[k]
            if gc == 0:
                continue
            i0 = max(0, -k)
            i1 = min(N, N - k)
            if i1 <= i0:
                continue
            g[i0:i1] += gc * f[i0 + k:i1 + k]
            g[i0:i1] += gc * f[i0 - k:i1 - k]
        f = np.maximum(f - lr * g, 0.0)
        lr *= 0.999
    return best_f, best_s


def construct():
    global START, DEADLINE, MARGIN
    START = time.time()
    DEADLINE = 120.0
    MARGIN = 12.0

    # fallback: all-ones (score 1.0)
    fallback = [1.0] * N

    # Phase 1: Gaussian superposition
    try:
        f, s = _phase1()
        f = np.maximum(f, 0.0)
        s = _score(f)
        if not np.isfinite(s) or s >= 1.0:
            f, s = np.ones(N), 1.0
    except Exception:
        f, s = np.ones(N), 1.0

    # Phase 2: direct bin-level polish
    try:
        if _time_left() > 6.0:
            f2, s2 = _phase2(f, s)
            if s2 < s:
                f, s = f2, s2
    except Exception:
        pass

    # Phase 3: high-precision polish
    try:
        if _time_left() > 3.0:
            f3, s3 = _phase3(f, s)
            if s3 < s:
                f, s = f3, s3
    except Exception:
        pass

    # final safety: ensure finite, nonnegative
    f = np.asarray(f, dtype=float)
    if f.shape != (N,):
        return fallback
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.maximum(f, 0.0)
    final_s = _score(f)
    if not np.isfinite(final_s) or final_s >= 1.0:
        return fallback
    return [float(x) for x in f]
