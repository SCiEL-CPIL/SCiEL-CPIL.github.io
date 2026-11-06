"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _value(h):
    """Compute the exact objective value (uses direct convolve)."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))


def _fft_conv_max(h):
    """Max of the full autocorrelation via FFT (fast, for optimization)."""
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * np.conj(f), size)
    return float(np.max(conv))


def _fast_value(h):
    """Fast objective using FFT for the autocorrelation peak."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e12
    m = _fft_conv_max(h)
    return abs(2.0 * N * m / (s * s))


def construct():
    global _t0
    _t0 = time.time()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    best_h = np.ones(N)
    best_v = _value(best_h)

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if len(h) != N or not np.all(np.isfinite(h)):
            return
        s = float(np.sum(h))
        if abs(s) < 1e-12:
            return
        v = _value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()

    # ---- Structural family 1: Gaussian FIR filter approximation ----
    # We model h[i] as a shifted and scaled Gaussian window.
    # h[i] = A * exp(-((i - mu)^2) / (2 * sigma^2))
    # Parameters: A (amplitude), mu (center), sigma (width).
    # We optimize these 3 parameters. To ensure sum(h) != 0 and handle scaling,
    # we fix A=1 and let the optimizer find the best mu and sigma.
    # We also try negative amplitude to explore symmetric minima if any.
    i = np.arange(N)
    
    def _gauss_recon(p):
        A, mu, sigma = p
        if sigma < 1e-6: sigma = 1e-6
        return A * np.exp(-((i - mu) ** 2) / (2 * sigma ** 2))

    def _obj_gauss(p):
        h = _gauss_recon(p)
        return _fast_value(h)

    rng = np.random.default_rng(12345)
    
    # Initial guesses for (A, mu, sigma)
    inits_gauss = []
    # Centered Gaussians with various widths
    for sig in [5, 10, 20, 50, 100]:
        inits_gauss.append([1.0, N / 2.0, sig])
        inits_gauss.append([-1.0, N / 2.0, sig])
    # Off-center Gaussians
    for mu in [N * 0.25, N * 0.75]:
        for sig in [10, 30]:
            inits_gauss.append([1.0, mu, sig])
            inits_gauss.append([-1.0, mu, sig])
    
    # Add some random inits
    for _ in range(5):
        A = rng.uniform(-2, 2)
        mu = rng.uniform(0, N)
        sig = rng.uniform(5, N)
        inits_gauss.append([A, mu, sig])

    bounds_gauss = [(-10, 10), (0, N), (1.0, N)]

    for init in inits_gauss:
        if not deadline_ok():
            break
        try:
            res = minimize(
                _obj_gauss, init, method='L-BFGS-B',
                bounds=bounds_gauss,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                update_best(_gauss_recon(res.x))
        except Exception:
            pass

    # ---- Structural family 2: localized Hamming atoms (current idea) ----
    atom_len = 10
    num_offsets = 15
    offsets = [o for o in range(0, N, N // num_offsets) if o + atom_len <= N]
    atom = np.hamming(atom_len)
    s_atom = atom.sum()
    if s_atom != 0:
        atom = atom / s_atom
    basis = []
    for sign in (1.0, -1.0):
        for off in offsets:
            v = np.zeros(N)
            v[off:off + atom_len] = sign * atom
            basis.append(v)
    basis_arr = np.array(basis)
    num_params = len(basis_arr)

    if num_params > 0:
        def _recon_atom(p):
            return p @ basis_arr

        def _obj_atom(p):
            return _fast_value(_recon_atom(p))

        inits_atom = [np.ones(num_params) / num_params]
        for _ in range(4):
            inits_atom.append(rng.standard_normal(num_params) * 0.1)
        for _ in range(3):
            init = np.zeros(num_params)
            idx = rng.choice(num_params, size=max(1, num_params // 4), replace=False)
            init[idx] = rng.standard_normal(len(idx)) * 0.5
            inits_atom.append(init)

        for init in inits_atom:
            if not deadline_ok():
                break
            try:
                res = minimize(
                    _obj_atom, init, method='L-BFGS-B',
                    bounds=[(-5, 5)] * num_params,
                    options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    update_best(_recon_atom(res.x))
            except Exception:
                pass

    # ---- Refinement: coordinate descent on the full 400-dim vector ----
    # Uses the fast FFT objective; a different search method (steepest-ish
    # coordinate descent) to polish whatever structural family found best.
    if deadline_ok():
        h = best_h.copy()
        cur = _fast_value(h)
        deltas = [0.05, -0.05, 0.2, -0.2, 0.8, -0.8]
        for _ in range(60):
            if not deadline_ok():
                break
            improved = False
            for idx in range(N):
                if not deadline_ok():
                    break
                old = h[idx]
                for d in deltas:
                    h[idx] = old + d
                    v = _fast_value(h)
                    if v < cur - 1e-12:
                        cur = v
                        improved = True
                        break
                h[idx] = old
            if not improved:
                break
        update_best(h)

    # ---- Final validation with the exact evaluator objective ----
    s = float(np.sum(best_h))
    if s == 0.0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    return np.ones(N)
# EVOLVE-BLOCK-END
