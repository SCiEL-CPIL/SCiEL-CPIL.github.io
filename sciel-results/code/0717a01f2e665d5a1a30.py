"""Second autocorrelation inequality: maximize ||f||2^2 / (||f||1 * ||f*f||inf)
over 50 nonnegative step heights.  DFT-parameterized spectral search + local polish."""

import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
_DEADLINE = time.time() + 124.0  # safety margin before the 150s wall-clock kill


def _autocorr_max(f):
    """Exact discrete autocorrelation supremum norm (full cross-correlation max)."""
    n = len(f)
    m = 2 * n - 1
    r = [0.0] * m
    for i in range(n):
        fi = f[i]
        if fi == 0.0:
            continue
        r[i] += fi * fi
        for j in range(1, n - i):
            r[i + j] += fi * f[i + j]
    return max(r)


def _R(f):
    """Exact objective: ||f||2^2 / (||f||1 * ||f*f||inf)."""
    s2 = sum(x * x for x in f)
    s1 = sum(f)
    if s1 <= 0.0:
        return 0.0
    pk = _autocorr_max(f)
    if pk <= 0.0:
        return 0.0
    return s2 / (s1 * pk)


def _R_smooth(f, s=0.25):
    """Smooth surrogate objective (max -> soft-max) for gradient-free polish."""
    n = len(f)
    m = 2 * n - 1
    r = [0.0] * m
    for i in range(n):
        fi = f[i]
        if fi == 0.0:
            continue
        r[i] += fi * fi
        for j in range(1, n - i):
            r[i + j] += fi * f[i + j]
    mx = max(r)
    if mx <= 0.0:
        return 0.0
    pk = mx + s * math.log(sum(math.exp((x - mx) / s) for x in r))
    s1 = sum(f)
    if s1 <= 0.0:
        return 0.0
    return sum(x * x for x in f) / (s1 * pk)


def _nelder_mead(fun, x0, iters, step):
    """Minimal Nelder-Mead; returns best x and best value."""
    d = len(x0)
    simplex = [[x0[i] for i in range(d)]]
    for i in range(d):
        p = list(x0)
        p[i] = x0[i] + step if x0[i] != 0 else step
        simplex.append(p)
    vals = [fun(p) for p in simplex]
    best_x = min(simplex, key=lambda p: fun(p))
    best_v = min(vals)
    for _ in range(iters):
        order = sorted(range(d + 1), key=lambda k: vals[k])
        simplex = [simplex[k] for k in order]
        vals = [vals[k] for k in order]
        if time.time() > _DEADLINE:
            break
        c = [sum(simplex[k][i] for k in range(d)) / d for i in range(d)]
        xr = [c[i] + 1.0 * (c[i] - simplex[-1][i]) for i in range(d)]
        fr = fun(xr)
        if fr < vals[0]:
            xe = [c[i] + 2.0 * (xr[i] - c[i]) for i in range(d)]
            fe = fun(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = [c[i] + 0.5 * (simplex[-1][i] - c[i]) for i in range(d)]
            fc = fun(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for k in range(1, d + 1):
                    simplex[k] = [simplex[0][i] + 0.5 * (simplex[k][i] - simplex[0][i])
                                  for i in range(d)]
                    vals[k] = fun(simplex[k])
        bi = min(range(d + 1), key=lambda k: vals[k])
        if vals[bi] < best_v:
            best_v = vals[bi]
            best_x = simplex[bi]
    return best_x, best_v


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    # --- DIRECT construction from structure ---------------------------------
    # f[0]=B, f[1..49]=b.  R(B,b)=(1+49 t^2)/((1+49 t) B), t=b/B.
    B, b = 2.0, 0.30
    f = [B] + [b] * (N - 1)
    best = _R(f)
    best_f = list(f)

    # --- coarse grid search over (B, b) -------------------------------------
    def _direct(B, b):
        ff = [B] + [b] * (N - 1)
        return _R(ff)

    grid_B = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    grid_b = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.20, 1.60]
    for B in grid_B:
        if time.time() > _DEADLINE:
            break
        for b in grid_b:
            v = _direct(B, b)
            if v > best:
                best, best_f = v, [B] + [b] * (N - 1)

    # --- light local polish (nonnegativity enforced) -------------------------
    try:
        f0 = [max(1e-6, x) for x in best_f]
        fbest, _ = _nelder_mead(_R_smooth, f0, iters=400, step=0.15)
        for cand in (fbest, f0):
            cand = [max(0.0, x) for x in cand]
            v = _R(cand)
            if v > best:
                best, best_f = v, cand
    except Exception:
        pass

    # --- DFT-parameterized spectral search -----------------------------------
    # h_k = |P(omega^k)|^2 where P is a polynomial with real coefficients.
    # This enforces non-negativity and leverages spectral structure.
    # We parameterize by the coefficients of P and search over them.

    def _from_coeffs(coeffs):
        """Given real coefficients of a polynomial P of degree d, compute h_k = |P(w^k)|^2
        for k=0..N-1 where w = exp(2*pi*i/N)."""
        n = len(coeffs)
        # Evaluate P at N roots of unity
        z = np.exp(2j * np.pi * np.arange(N) / N)
        # P(z) = sum_j coeffs[j] * z^j
        Pvals = np.zeros(N, dtype=complex)
        for j, c in enumerate(coeffs):
            Pvals += c * (z ** j)
        h = np.abs(Pvals) ** 2
        # Normalize: we want nonnegative heights; scale doesn't matter for ratio
        return h

    def _ratio_from_h(h):
        """Compute the objective ratio from a nonnegative array h."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        s2 = np.sum(h ** 2)
        s1 = np.sum(h)
        if s1 <= 0:
            return 0.0
        # Autoconvolution: (h*h)[k] for k in -(N-1)..(N-1)
        # Using FFT for speed
        n = len(h)
        m = 2 * n - 1
        # Pad to length >= m for linear convolution
        size = 1
        while size < m:
            size *= 2
        H = np.fft.rfft(h, size)
        conv = np.fft.irfft(H * np.conj(H), size)[:m]
        pk = np.max(conv)
        if pk <= 0:
            return 0.0
        return s2 / (s1 * pk)

    def _objective(coeffs):
        """Objective for DFT parameterization: minimize negative ratio."""
        if time.time() > _DEADLINE:
            return 0.0
        h = _from_coeffs(coeffs)
        r = _ratio_from_h(h)
        return -r

    # Search over polynomial coefficients. Start with degree 5 (6 coefficients).
    # Bounds: coefficients in [-1, 1] initially, we can scale.
    bounds = [(-1.0, 1.0)] * 6
    x0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # P(z) = 1 initially

    # Try differential evolution for global search
    try:
        if time.time() < _DEADLINE - 30:
            res = differential_evolution(
                _objective, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-6, polish=True, init='latinhypercube'
            )
            if res.success or res.fun < 0:
                h_candidate = _from_coeffs(res.x)
                v = _ratio_from_h(h_candidate)
                if v > best:
                    best = v
                    best_f = list(h_candidate)
    except Exception:
        pass

    # Try L-BFGS-B from multiple starting points
    try:
        if time.time() < _DEADLINE - 20:
            # Several diverse starting points
            starts = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [1.0, 0.3, 0.3, 0.0, 0.0, 0.0],
                [0.8, 0.4, 0.2, 0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                [1.0, 0.2, 0.2, 0.2, 0.2, 0.0],
            ]
            for x0 in starts:
                if time.time() > _DEADLINE - 10:
                    break
                try:
                    res = minimize(_objective, x0, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': 200})
                    if res.fun < 0:
                        h_candidate = _from_coeffs(res.x)
                        v = _ratio_from_h(h_candidate)
                        if v > best:
                            best = v
                            best_f = list(h_candidate)
                except Exception:
                    pass
    except Exception:
        pass

    # --- Final local polish on best_f using Nelder-Mead ----------------------
    try:
        if time.time() < _DEADLINE - 5:
            f0 = [max(1e-6, x) for x in best_f]
            fbest, _ = _nelder_mead(_R_smooth, f0, iters=300, step=0.1)
            cand = [max(0.0, x) for x in fbest]
            v = _R(cand)
            if v > best:
                best, best_f = v, cand
    except Exception:
        pass

    # --- guaranteed fallback -------------------------------------------------
    if best <= 1.0:
        best = 1.0
        best_f = [1.0] * N

    return best_f
# EVOLVE-BLOCK-END
