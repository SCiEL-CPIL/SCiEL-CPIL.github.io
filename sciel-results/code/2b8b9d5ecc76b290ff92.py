import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_i = np.arange(N, dtype=float)
_t0 = time.time()
_DEADLINE = 138.0

# Atom-based sparse parameterization
_ATOM_LEN = 16
_atom_base = np.hamming(_ATOM_LEN)
_atom_base = _atom_base / np.sum(_atom_base)  # normalize so sum=1
_MAX_ATOMS = 8  # max number of atoms in sparse representation


def _ok():
    return (time.time() - _t0) < _DEADLINE


def _value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _fft_conv(h):
    size = 2 * N
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)[:2 * N - 1]


def _smooth_value(h, beta=100.0):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = _fft_conv(h)
    m = np.max(conv)
    scaled = np.clip(beta * (conv - m), -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * lse / (s * s))


# DCT basis for spectral shaping
_K_DCT = 40
_n = np.arange(N)
_k = np.arange(_K_DCT)
_angles = np.pi * (2 * _n[:, None] + 1) * _k[None, :] / (2 * N)
_DCT_BASIS = np.cos(_angles)
_norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
_norms[_norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / _norms
_DCT_BASIS_T = _DCT_BASIS.T


def _dct_to_h(coeffs):
    return _DCT_BASIS @ coeffs


def _dct_smooth_obj(coeffs, beta=100.0):
    h = _dct_to_h(coeffs)
    return _smooth_value(h, beta)


def _dct_exact_obj(coeffs):
    h = _dct_to_h(coeffs)
    return _value(h)


def _atoms_to_h(params):
    """Convert atom parameters to full h vector.
    params: array of shape (2*MAX_ATOMS,) = [amp1, off1, amp2, off2, ...]
    Only first _MAX_ATOMS atoms are used.
    """
    h = np.zeros(N)
    for k in range(_MAX_ATOMS):
        amp = params[2 * k]
        off = int(round(params[2 * k + 1]))
        if abs(amp) < 1e-10:
            continue
        # Place atom at offset off, clipped to [0, N)
        start = off
        end = off + _ATOM_LEN
        if start >= N or end <= 0:
            continue
        s = max(0, start)
        e = min(N, end)
        if s < e:
            h[s:e] += amp * _atom_base[s - start : e - start]
    return h


def _atoms_smooth_obj(params, beta=100.0):
    h = _atoms_to_h(params)
    return _smooth_value(h, beta)


def _atoms_exact_obj(params):
    h = _atoms_to_h(params)
    return _value(h)


def _atom_search():
    """Optimize sparse atom representation."""
    rng = np.random.default_rng(777)
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    
    # Try different numbers of atoms
    for n_atoms in [2, 3, 4, 5, 6, _MAX_ATOMS]:
        if not _ok():
            break
        for trial in range(3):
            if not _ok():
                break
            # Initialize: symmetric pairs of atoms
            params = np.zeros(2 * _MAX_ATOMS)
            for k in range(n_atoms):
                # Alternating signs, symmetric offsets
                amp = 1.0 if k % 2 == 0 else -1.0
                # Spread offsets across the domain
                if n_atoms == 1:
                    off = N // 2 - _ATOM_LEN // 2
                else:
                    frac = k / (n_atoms - 1) if n_atoms > 1 else 0.5
                    off = int(frac * (N - _ATOM_LEN))
                params[2 * k] = amp * rng.uniform(0.5, 1.5)
                params[2 * k + 1] = off + rng.integers(-5, 5)
            
            try:
                res = minimize(_atoms_smooth_obj, params, method='L-BFGS-B',
                              args=(80.0,),
                              bounds=[(-3, 3)] * n_atoms + [(0, N - _ATOM_LEN)] * n_atoms + [(0, 3)] * (_MAX_ATOMS - n_atoms) + [(0, N)] * (_MAX_ATOMS - n_atoms),
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.x is not None:
                    h = _atoms_to_h(res.x)
                    update(h)
            except Exception:
                pass
    
    # Polish with coordinate descent
    if _ok():
        try:
            h_cd, _ = _coord_descent(best_h, iters=10, beta=100.0, step=0.02)
            update(h_cd)
        except Exception:
            pass
    
    return best_h


def _coord_descent(h0, iters=20, beta=100.0, step=0.05):
    h = h0.copy()
    cur = _smooth_value(h, beta)
    rng = np.random.default_rng(42)
    for it in range(iters):
        if not _ok():
            break
        improved = False
        order = rng.permutation(N)
        for idx in order:
            for d in (step, -step):
                h2 = h.copy()
                h2[idx] += d
                v2 = _smooth_value(h2, beta)
                if v2 < cur:
                    h = h2
                    cur = v2
                    improved = True
                    break
        if not improved:
            step *= 0.6
            if step < 1e-5:
                break
    return h, cur


def construct():
    global _t0
    _t0 = time.time()
    rng = np.random.default_rng(2024)

    best_h = np.ones(N)
    best_v = _value(best_h)

    def update(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if _ok():
        # Initial DCT coefficient vectors from diverse starting points
        init_pool = []
        
        # Start 1: All zeros (flat signal)
        init_pool.append(np.zeros(_K_DCT))
        
        # Start 2: Low-frequency emphasis
        c2 = np.zeros(_K_DCT)
        c2[0] = 1.0
        init_pool.append(c2)
        
        # Start 3: Random low-frequency
        for trial in range(10):
            if not _ok():
                break
            c = rng.normal(0, 0.5, _K_DCT)
            # Emphasize low frequencies
            weights = np.linspace(1.0, 0.1, _K_DCT)
            c *= weights
            init_pool.append(c)
        
        # Start 4: Chirp-like DCT spectrum
        for rate in [-30, -15, 15, 30]:
            if not _ok():
                break
            c = np.zeros(_K_DCT)
            for j in range(_K_DCT):
                c[j] = np.cos(rate * j * j / _K_DCT) * (1.0 / (1.0 + j))
            init_pool.append(c)
        
        # Optimize each starting point with L-BFGS-B
        for c0 in init_pool:
            if not _ok():
                break
            try:
                res = minimize(_dct_smooth_obj, c0, method='L-BFGS-B',
                              args=(100.0,),
                              options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _dct_to_h(res.x)
                    update(h)
            except Exception:
                pass

    # Phase 2: Full-dimensional coordinate descent on best candidates
    if _ok():
        # Also try coordinate descent from the current best
        h_cd, _ = _coord_descent(best_h, iters=15, beta=100.0, step=0.03)
        update(h_cd)

    # Phase 3: DCT optimization with different beta values (annealing)
    if _ok():
        # Get current best in DCT coefficients
        h_best = best_h
        c_best = _DCT_BASIS_T @ h_best
        
        for beta in [50.0, 80.0, 120.0, 200.0]:
            if not _ok():
                break
            try:
                res = minimize(_dct_smooth_obj, c_best, method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _dct_to_h(res.x)
                    update(h)
                    c_best = res.x
            except Exception:
                pass

    # Phase 4: Additional DCT restarts with perturbation
    if _ok():
        for trial in range(5):
            if not _ok():
                break
            c_pert = c_best + rng.normal(0, 0.3, _K_DCT)
            try:
                res = minimize(_dct_smooth_obj, c_pert, method='L-BFGS-B',
                              args=(100.0,),
                              options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _dct_to_h(res.x)
                    update(h)
            except Exception:
                pass

    # Phase 5: Final coordinate descent polish
    if _ok():
        try:
            h_fin, _ = _coord_descent(best_h, iters=10, beta=150.0, step=0.02)
            update(h_fin)
        except Exception:
            pass

    # Phase 6: Nelder-Mead on DCT coefficients for final polish
    if _ok():
        try:
            c_best = _DCT_BASIS_T @ best_h
            res = minimize(_dct_smooth_obj, c_best, method='Nelder-Mead',
                          args=(100.0,),
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
            if res.x is not None:
                h = _dct_to_h(res.x)
                update(h)
        except Exception:
            pass

    # Phase 7: Atom-based sparse parameterization search
    if _ok():
        try:
            h_atom = _atom_search()
            update(h_atom)
        except Exception:
            pass

    # Phase 8: Final polish with tighter tolerances
    if _ok():
        try:
            c_final = _DCT_BASIS_T @ best_h
            res = minimize(_dct_smooth_obj, c_final, method='L-BFGS-B',
                          args=(200.0,),
                          options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.x is not None:
                h = _dct_to_h(res.x)
                update(h)
        except Exception:
            pass

    # Phase 9: Very fine coordinate descent
    if _ok():
        try:
            h_fin2, _ = _coord_descent(best_h, iters=8, beta=200.0, step=0.01)
            update(h_fin2)
        except Exception:
            pass

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
