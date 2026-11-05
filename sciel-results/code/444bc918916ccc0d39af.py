"""
Third autocorrelation inequality:
    value = |2*N*max(conv(h,h)) / sum(h)^2|
Scale-invariant in h. We normalize h to unit L2 norm, so the objective becomes
    |2*N*max(conv(h,h)) / (sum(h)^2)|  with ||h||_2 = 1.

Strategy:
- Parameterize h by a low-dimensional DCT coefficient vector (real, band-limited shape).
- Global search: differential evolution on the coefficients.
- Local refinement: L-BFGS-B with a smooth soft-max surrogate (fast, via FFT conv).
- Final polish: L-BFGS-B directly on h with the soft-max surrogate.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400
DEADLINE = 135.0  # safety margin before the 150s kill


def _fft_convolve(h):
    """Full linear autoconvolution via FFT (faster than np.convolve for N=400)."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)[:2 * n - 1]


def _value(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _smooth_max_value(h, beta):
    """Differentiable surrogate: soft-max of conv entries, scale-invariant form."""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    c = conv / beta
    cmax = np.max(c)
    lse = cmax + np.log(np.sum(np.exp(c - cmax)))
    smax = beta * lse
    return abs(2.0 * N * smax / (s * s))


# --- DCT (orthonormal, type-II) basis for real band-limited shape ---
_t = np.arange(N, dtype=float)
_dct_basis = np.empty((N, N))
for k in range(N):
    _dct_basis[:, k] = np.cos(np.pi * k * (2 * _t + 1) / (2.0 * N))
_dct_basis *= np.sqrt(2.0 / N)  # orthonormal


def _dct_h(c):
    """Reconstruct h from DCT coefficients (unit-norm shape)."""
    h = _dct_basis @ np.asarray(c, dtype=float)
    n = np.linalg.norm(h)
    if n < 1e-12:
        h = np.ones(N)
        n = np.sqrt(N)
    return h / n


def _dct_obj(c):
    return _value(_dct_h(c))


def _dct_smooth_obj(c, beta):
    return _smooth_max_value(_dct_h(c), beta)


def construct():
    t_start = time.time()

    def remaining():
        return DEADLINE - (time.time() - t_start)

    # Seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    # DCT coefficient space: use a modest number of low-frequency coefficients.
    M = 40
    rng = np.random.default_rng(7)

    # --- Phase 1: Differential evolution on DCT coefficients ---
    if remaining() > 35.0:
        bounds = [(-1.0, 1.0)] * M
        try:
            res = differential_evolution(
                _dct_obj,
                bounds,
                seed=42,
                maxiter=200,
                popsize=10,
                tol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = _dct_h(res.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart L-BFGS-B (soft-max) on DCT coefficients ---
    n_restarts = 14
    for i in range(n_restarts):
        if remaining() < 25.0:
            break
        # Starting point: random, or perturbed best
        if i == 0:
            c0 = np.zeros(M)
            c0[0] = 1.0  # dominant DC
        elif i == 1:
            c0 = rng.normal(0, 0.3, M)
        else:
            c0 = rng.normal(0, 0.5, M)
        try:
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=30.0),
                c0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11, 'iprint': 0},
            )
            if res.fun < np.inf:
                h_cand = _dct_h(res.x)
                v_cand = _value(h_cand)
                if v_cand < best_val:
                    best_val = v_cand
                    best_h = h_cand
        except Exception:
            pass

    # --- Phase 3: Direct L-BFGS-B on h with soft-max surrogate ---
    if remaining() > 20.0:
        try:
            res = minimize(
                lambda hv: _smooth_max_value(hv, beta=15.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0},
            )
            if res.fun < np.inf:
                h_final = res.x
                v_final = _value(h_final)
                if v_final < best_val and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_val = v_final
                    best_h = h_final
        except Exception:
            pass

    # --- Phase 4: Nelder-Mead on DCT coefficients for a final local nudge ---
    if remaining() > 12.0:
        # Re-extract approximate DCT coefficients from best_h for a local start
        c_start = _dct_basis.T @ best_h
        c_start = c_start[:M]
        try:
            res = minimize(
                _dct_obj,
                c_start,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
            )
            if res.fun < best_val:
                h_cand = _dct_h(res.x)
                v_cand = _value(h_cand)
                if v_cand < best_val:
                    best_val = v_cand
                    best_h = h_cand
        except Exception:
            pass

    # Validity guard
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
