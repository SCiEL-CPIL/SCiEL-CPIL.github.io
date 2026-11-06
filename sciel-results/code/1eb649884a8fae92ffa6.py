import time
import numpy as np
from scipy.optimize import minimize

N = 400
_i = np.arange(N, dtype=float)
_t0 = time.time()
_DEADLINE = 145.0

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
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * lse / (s * s))

# Precompute DCT basis
_K_DCT = 80
_n = np.arange(N)
_k = np.arange(_K_DCT)
angles = np.pi * (2 * _n[:, None] + 1) * _k[None, :] / (2 * N)
_DCT_BASIS = np.cos(angles)
norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
norms[norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / norms

def _dct_h(coeffs):
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=100.0):
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

# Atom-based parameterization
_ATOM_LEN = 12
_atom_template = np.hamming(_ATOM_LEN)
_atom_template = _atom_template / np.sum(_atom_template)  # normalize sum to 1

def _atom_h(params):
    n_atoms = len(params) // 2
    amps = params[:n_atoms]
    offsets = params[n_atoms:]
    h = np.zeros(N)
    for i in range(n_atoms):
        off = int(np.round(offsets[i]))
        off = off % N
        start = off
        end = start + _ATOM_LEN
        if end <= N:
            h[start:end] += amps[i] * _atom_template
        else:
            first_part = N - start
            h[start:] += amps[i] * _atom_template[:first_part]
            h[:end - N] += amps[i] * _atom_template[first_part:]
    return h

def _atom_smooth_obj(params, beta=100.0):
    h = _atom_h(params)
    return _smooth_value(h, beta)

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

    # Phase 1: DCT-based optimization with higher beta for better surrogate fidelity
    num_restarts = 15
    for r in range(num_restarts):
        if not _ok():
            break
            
        init_coeffs = rng.normal(0, 0.15, _K_DCT)
        init_coeffs[0] = rng.uniform(0.5, 2.0)
        
        try:
            res = minimize(
                _dct_smooth_obj, 
                init_coeffs, 
                method='L-BFGS-B', 
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                h_candidate = _dct_h(res.x)
                update(h_candidate)
        except Exception:
            pass

    # Phase 2: Atom-based optimization - sparse structure for peak suppression
    if _ok():
        n_atoms = 10
        num_atom_restarts = 12
        for r in range(num_atom_restarts):
            if not _ok():
                break
            
            amps = rng.uniform(-1, 1, n_atoms)
            offsets = rng.uniform(0, N, n_atoms)
            params = np.concatenate([amps, offsets])
            
            try:
                res = minimize(
                    _atom_smooth_obj, 
                    params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.x is not None:
                    h_candidate = _atom_h(res.x)
                    update(h_candidate)
            except Exception:
                pass

    # Phase 3: Full-dimensional coordinate descent with in-place updates
    if _ok():
        h_polish = best_h.copy()
        cur_v = _value(h_polish)
        step_size = 0.05
        max_iters = 100
        
        for it in range(max_iters):
            if not _ok():
                break
            improved = False
            order = rng.permutation(N)
            for idx in order:
                orig = h_polish[idx]
                for d in (step_size, -step_size):
                    h_polish[idx] = orig + d
                    v2 = _value(h_polish)
                    if v2 < cur_v:
                        cur_v = v2
                        improved = True
                        break
                else:
                    h_polish[idx] = orig
            if not improved:
                step_size *= 0.7
                if step_size < 1e-4:
                    break
        update(h_polish)

    # Phase 4: Nelder-Mead polish on DCT subspace to escape local minima
    if _ok():
        # Project best_h onto DCT subspace
        h_proj = _DCT_BASIS @ (_DCT_BASIS.T @ best_h)
        init_coeffs = _DCT_BASIS.T @ h_proj
        
        try:
            res = minimize(
                _dct_smooth_obj,
                init_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.x is not None:
                h_candidate = _dct_h(res.x)
                update(h_candidate)
        except Exception:
            pass

    # Phase 5: Quick check on simple chirps if time remains
    if _ok():
        for rate in [-50, -20, 20, 50]:
            for amp in [0.5, 1.0]:
                for base in [0.0, 0.5, -0.5]:
                    h_chirp = base + amp * np.cos(rate * (_i / N) ** 2)
                    update(h_chirp)
                    if not _ok():
                        break

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
