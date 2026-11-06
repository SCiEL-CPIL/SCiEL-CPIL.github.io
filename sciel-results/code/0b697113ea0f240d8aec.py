"""
Sparse atom-based step height construction for third autocorrelation inequality.

Strategy:
- Model h as a sparse linear combination of shifted, scaled, and sign-flipped copies
  of a single short "atom" (length-10 Hamming window).
- Parameterize by a small set of amplitudes and integer offsets.
- Use differential evolution + local refinement to optimize the low-dimensional parameters.
- The structural symmetry minimizes peak autocorrelation.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.signal import fftconvolve

N = 400
_t0 = time.time()
_DEADLINE = 135.0

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

# Atom definition: length-10 Hamming window
_ATOM_LEN = 10
_atom = np.hamming(_ATOM_LEN)

def _build_h(params):
    """
    Build h from sparse atom parameters.
    params: [amp_0, offset_0, sign_0, amp_1, offset_1, sign_1, ..., amp_K-1, offset_K-1, sign_K-1, base]
    K atoms, each with amplitude, integer offset, and sign.
    """
    K = (len(params) - 1) // 3
    h = np.zeros(N)
    for k in range(K):
        amp = params[3 * k]
        offset = int(round(params[3 * k + 1]))
        sign = 1.0 if params[3 * k + 2] >= 0 else -1.0
        # Place atom at offset, clip to valid range
        start = offset
        end = offset + _ATOM_LEN
        if start < 0:
            atom_slice = _atom[-start:]
            h[:end] += sign * amp * atom_slice
        elif end > N:
            atom_slice = _atom[:N - start]
            h[start:N] += sign * amp * atom_slice
        else:
            h[start:end] += sign * amp * _atom
    base = params[-1]
    h += base
    return h

def _obj(params):
    """Exact objective."""
    h = _build_h(params)
    return _value(h)

def _smooth_obj(params, beta=80.0):
    """Smooth objective."""
    h = _build_h(params)
    return _smooth_max_value(h, beta)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Differential evolution with 4 atoms
    if deadline_ok():
        try:
            K = 4
            n_params = 3 * K + 1
            bounds = []
            for k in range(K):
                bounds.append((-3.0, 3.0))   # amplitude
                bounds.append((0.0, float(N - _ATOM_LEN)))  # offset
                bounds.append((-1.0, 1.0))   # sign
            bounds.append((-2.0, 2.0))       # base
            
            res = differential_evolution(
                _obj, bounds, seed=42, maxiter=80, tol=1e-8,
                popsize=15, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Differential evolution with 6 atoms
    if deadline_ok():
        try:
            K = 6
            n_params = 3 * K + 1
            bounds = []
            for k in range(K):
                bounds.append((-3.0, 3.0))
                bounds.append((0.0, float(N - _ATOM_LEN)))
                bounds.append((-1.0, 1.0))
            bounds.append((-2.0, 2.0))
            
            res = differential_evolution(
                _obj, bounds, seed=7, maxiter=60, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Differential evolution with 3 atoms (simpler, may find good symmetric structures)
    if deadline_ok():
        try:
            K = 3
            n_params = 3 * K + 1
            bounds = []
            for k in range(K):
                bounds.append((-3.0, 3.0))
                bounds.append((0.0, float(N - _ATOM_LEN)))
                bounds.append((-1.0, 1.0))
            bounds.append((-2.0, 2.0))
            
            res = differential_evolution(
                _obj, bounds, seed=13, maxiter=100, tol=1e-8,
                popsize=20, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Local refinement of best solution using L-BFGS-B on continuous parameters
    if deadline_ok():
        try:
            # Project best_h back to atom parameters via least squares
            # We'll use a simpler approach: try multiple initializations and refine
            for K in [3, 4, 5]:
                if not deadline_ok():
                    break
                n_params = 3 * K + 1
                # Try to fit best_h with K atoms
                # Initialize with evenly spaced offsets
                offsets = np.linspace(0, N - _ATOM_LEN, K, dtype=int)
                amps = np.full(K, 1.0)
                signs = np.ones(K)
                base = float(np.mean(best_h))
                
                x0 = []
                for k in range(K):
                    x0.extend([amps[k], float(offsets[k]), signs[k]])
                x0.append(base)
                
                res = minimize(
                    lambda x: _smooth_obj(x, beta=100.0),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    h = _build_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 5: Full-dimensional coordinate descent polish
    if deadline_ok():
        try:
            h_cd = best_h.copy()
            best_v_cd = _smooth_max_value(h_cd, beta=80.0)
            step_size = 0.01
            for it in range(100):
                if not deadline_ok():
                    break
                improved = False
                for i in range(N):
                    for delta in [step_size, -step_size]:
                        h_trial = h_cd.copy()
                        h_trial[i] += delta
                        v_trial = _smooth_max_value(h_trial, beta=80.0)
                        if v_trial < best_v_cd:
                            h_cd = h_trial
                            best_v_cd = v_trial
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 6: Try a different atom shape (length 8) for diversity
    if deadline_ok():
        try:
            _atom8 = np.hamming(8)
            def _build_h8(params):
                K = (len(params) - 1) // 3
                h = np.zeros(N)
                for k in range(K):
                    amp = params[3 * k]
                    offset = int(round(params[3 * k + 1]))
                    sign = 1.0 if params[3 * k + 2] >= 0 else -1.0
                    start = offset
                    end = offset + 8
                    if start < 0:
                        atom_slice = _atom8[-start:]
                        h[:end] += sign * amp * atom_slice
                    elif end > N:
                        atom_slice = _atom8[:N - start]
                        h[start:N] += sign * amp * atom_slice
                    else:
                        h[start:end] += sign * amp * _atom8
                base = params[-1]
                h += base
                return h
            
            def _obj8(params):
                h = _build_h8(params)
                return _value(h)
            
            K = 4
            bounds = []
            for k in range(K):
                bounds.append((-3.0, 3.0))
                bounds.append((0.0, float(N - 8)))
                bounds.append((-1.0, 1.0))
            bounds.append((-2.0, 2.0))
            
            res = differential_evolution(
                _obj8, bounds, seed=99, maxiter=60, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h8(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
