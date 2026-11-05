"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Use FFT-based fast convolution for speed (O(N log N) vs O(N^2)).
- DCT spectral representation: optimize in DCT coefficient space (400 dims).
- Multi-stage: coarse random search -> Powell refinement -> L-BFGS-B with soft-max.
- Deadline guard: return best before 150s wall-clock.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct
import time

N = 400
_conv_size = 2 * N - 1  # full convolution length

def _value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    # Fast full convolution via FFT
    f = np.fft.rfft(h, n=_conv_size)
    conv = np.fft.irfft(f * f, n=_conv_size)
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _smooth_max_value(h, beta=30.0):
    """Smooth surrogate using soft-max on the convolution."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    f = np.fft.rfft(h, n=_conv_size)
    conv = np.fft.irfft(f * f, n=_conv_size)
    # Soft max
    conv_scaled = conv / beta
    m = np.max(conv_scaled)
    log_sum_exp = m + np.log(np.sum(np.exp(conv_scaled - m)))
    soft_max_conv = beta * log_sum_exp
    return abs(2.0 * N * soft_max_conv / (s ** 2))

def _dct_h(c):
    """Convert DCT coefficients to h using scipy idct."""
    c = np.asarray(c, dtype=float)
    h = scipy_idct(c, type=2, norm='ortho')
    return h

def _dct_smooth_obj(c, beta=30.0):
    h = _dct_h(c)
    return _smooth_max_value(h, beta=beta)

def _dct_obj(c):
    h = _dct_h(c)
    return _value(h)

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0

    def remaining():
        return DEADLINE - time.time()

    # Initial best: seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    # Convert seed to DCT coefficients for optimization
    seed_c = scipy_dct(best_h, type=2, norm='ortho')

    # --- Phase 1: Random restarts in DCT space with Powell refinement ---
    rng = np.random.default_rng(42)
    n_restarts = 8
    for i in range(n_restarts):
        if remaining() < 20.0:
            break

        # Random DCT coefficients: low frequencies dominant
        freqs = np.arange(N, dtype=float)
        scale = 1.0 / (1.0 + 0.01 * freqs)
        c_rand = rng.standard_normal(N) * scale * 5.0
        c_rand[0] = rng.uniform(-5.0, 5.0)

        # Powell refinement
        try:
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=50.0),
                c_rand,
                method='Powell',
                options={'maxiter': 50, 'ftol': 1e-10, 'xtol': 1e-8, 'disp': False},
            )
            if res.fun < np.inf:
                h_check = _dct_h(res.x)
                v_check = _value(h_check)
                if v_check < best_val and np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-12:
                    best_val = v_check
                    best_h = h_check
        except Exception:
            pass

        # Also try L-BFGS-B from this point
        if remaining() > 25.0:
            try:
                start_c = res.x if hasattr(res, 'x') and res.x is not None else c_rand
                res2 = minimize(
                    lambda c: _dct_smooth_obj(c, beta=30.0),
                    start_c,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0},
                )
                if res2.fun < np.inf:
                    h_check2 = _dct_h(res2.x)
                    v_check2 = _value(h_check2)
                    if v_check2 < best_val and np.all(np.isfinite(h_check2)) and abs(np.sum(h_check2)) > 1e-12:
                        best_val = v_check2
                        best_h = h_check2
            except Exception:
                pass

    # --- Phase 2: L-BFGS-B refinement from current best in DCT space ---
    if remaining() > 15.0:
        try:
            best_c = scipy_dct(best_h, type=2, norm='ortho')
            res_final = minimize(
                lambda c: _dct_smooth_obj(c, beta=20.0),
                best_c,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0},
            )
            if res_final.fun < np.inf:
                h_final = _dct_h(res_final.x)
                v_final = _value(h_final)
                if v_final < best_val and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_val = v_final
                    best_h = h_final
        except Exception:
            pass

    # --- Phase 3: Direct h-space L-BFGS-B polish ---
    if remaining() > 10.0:
        try:
            res_direct = minimize(
                lambda hv: _smooth_max_value(hv, beta=15.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0},
            )
            if res_direct.fun < np.inf:
                h_direct = res_direct.x
                v_direct = _value(h_direct)
                if v_direct < best_val and np.all(np.isfinite(h_direct)) and abs(np.sum(h_direct)) > 1e-12:
                    best_val = v_direct
                    best_h = h_direct
        except Exception:
            pass

    # --- Phase 4: Coordinate descent on h (last resort if time remains) ---
    if remaining() > 5.0:
        try:
            h_cd = best_h.copy()
            n_cd_iters = 3
            for it in range(n_cd_iters):
                if remaining() < 3.0:
                    break
                for j in range(N):
                    if remaining() < 2.0:
                        break
                    for delta in [0.01, -0.01, 0.1, -0.1]:
                        h_trial = h_cd.copy()
                        h_trial[j] += delta
                        v_trial = _value(h_trial)
                        if v_trial < _value(h_cd) and np.all(np.isfinite(h_trial)) and abs(np.sum(h_trial)) > 1e-12:
                            h_cd = h_trial
                            break
            v_cd = _value(h_cd)
            if v_cd < best_val and np.all(np.isfinite(h_cd)) and abs(np.sum(h_cd)) > 1e-12:
                best_val = v_cd
                best_h = h_cd
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    # Final sanity check
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)

    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
