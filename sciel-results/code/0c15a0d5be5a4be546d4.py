"""
Sparse atom-based step height construction for third autocorrelation inequality.

Key insight: value = 2*N*max(conv(h_norm)) where h_norm = h/sum(h), so we optimize
only the shape. Use a short atom (length 10) with many placements and sign flips.
Optimize low-dimensional parameters: amplitudes, integer offsets, signs.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400
_t0 = time.time()
_DEADLINE = 135.0

# Atom: length-10 Hamming window
_ATOM_LEN = 10
_atom = np.hamming(_ATOM_LEN)

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _value_normalized(h):
    """Value depends only on normalized shape: 2*N*max(conv(h/sum(h)))."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h / s
    conv = _fft_convolve(h_norm)
    m = float(np.max(conv))
    return abs(2.0 * N * m)

def _build_h_from_atoms(amps, offsets, signs):
    """Build h from atom parameters using integer offsets."""
    h = np.zeros(N)
    for amp, off, sign in zip(amps, offsets, signs):
        start = int(round(off))
        if start < 0 or start + _ATOM_LEN > N:
            continue
        h[start:start + _ATOM_LEN] += sign * amp * _atom
    return h

def _obj_atom(params):
    """Objective: minimize 2*N*max(conv(h_norm)) for atom-based h."""
    K = len(params) // 3
    amps = params[0:3*K:3]
    offsets = params[1:3*K:3]
    signs = params[2:3*K:3]
    signs = np.where(signs >= 0, 1.0, -1.0)
    h = _build_h_from_atoms(amps, offsets, signs)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e6
    return _value_normalized(h)

def _obj_sym(params):
    """Symmetric objective: pairs of atoms with opposite signs at mirrored positions."""
    K_pairs = len(params) // 2
    h = np.zeros(N)
    for k in range(K_pairs):
        amp = params[2*k]
        offset = params[2*k + 1]
        start1 = int(round(offset))
        start2 = N - _ATOM_LEN - int(round(offset))
        if 0 <= start1 and start1 + _ATOM_LEN <= N:
            h[start1:start1 + _ATOM_LEN] += amp * _atom
        if 0 <= start2 and start2 + _ATOM_LEN <= N:
            h[start2:start2 + _ATOM_LEN] -= amp * _atom
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e6
    return _value_normalized(h)

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
        s = np.sum(h)
        if abs(s) < 1e-12:
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DE with 12 atoms (amp, offset, sign)
    if deadline_ok():
        try:
            K = 12
            bounds = []
            for _ in range(K):
                bounds.append((-1.0, 1.0))
                bounds.append((0.0, float(N - _ATOM_LEN)))
                bounds.append((-1.0, 1.0))
            
            res = differential_evolution(
                _obj_atom, bounds, seed=42, maxiter=150, tol=1e-10,
                popsize=25, init='sobol', polish=False
            )
            if res.x is not None and res.fun < best_v:
                K2 = len(res.x) // 3
                amps = res.x[0:3*K2:3]
                offsets = res.x[1:3*K2:3]
                signs = np.where(res.x[2:3*K2:3] >= 0, 1.0, -1.0)
                h = _build_h_from_atoms(amps, offsets, signs)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: DE with 16 atoms
    if deadline_ok():
        try:
            K = 16
            bounds = []
            for _ in range(K):
                bounds.append((-1.0, 1.0))
                bounds.append((0.0, float(N - _ATOM_LEN)))
                bounds.append((-1.0, 1.0))
            
            res = differential_evolution(
                _obj_atom, bounds, seed=7, maxiter=100, tol=1e-10,
                popsize=20, init='sobol', polish=False
            )
            if res.x is not None and res.fun < best_v:
                K2 = len(res.x) // 3
                amps = res.x[0:3*K2:3]
                offsets = res.x[1:3*K2:3]
                signs = np.where(res.x[2:3*K2:3] >= 0, 1.0, -1.0)
                h = _build_h_from_atoms(amps, offsets, signs)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Symmetric pairs (8 pairs = 16 atoms effectively)
    if deadline_ok():
        try:
            K_pairs = 8
            bounds = [(-1.0, 1.0), (0.0, float(N - _ATOM_LEN))] * K_pairs
            
            res = differential_evolution(
                _obj_sym, bounds, seed=13, maxiter=150, tol=1e-10,
                popsize=25, init='sobol', polish=False
            )
            if res.x is not None and res.fun < best_v:
                h = np.zeros(N)
                for k in range(K_pairs):
                    amp = res.x[2*k]
                    offset = res.x[2*k + 1]
                    start1 = int(round(offset))
                    start2 = N - _ATOM_LEN - int(round(offset))
                    if 0 <= start1 and start1 + _ATOM_LEN <= N:
                        h[start1:start1 + _ATOM_LEN] += amp * _atom
                    if 0 <= start2 and start2 + _ATOM_LEN <= N:
                        h[start2:start2 + _ATOM_LEN] -= amp * _atom
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Local refinement via gradient descent on full vector
    if deadline_ok():
        try:
            h_work = best_h.copy()
            s_init = np.sum(h_work)
            if abs(s_init) < 1e-12:
                h_work = np.ones(N)
                s_init = N
            else:
                h_work = h_work / s_init
            
            for it in range(100):
                if not deadline_ok():
                    break
                conv = _fft_convolve(h_work)
                peak_lag = int(np.argmax(conv))
                
                # Gradient of conv[peak_lag] w.r.t. h
                grad = np.zeros(N)
                for j in range(N):
                    if 0 <= j + peak_lag < N:
                        grad[j] += h_work[j + peak_lag]
                    if 0 <= j - peak_lag < N:
                        grad[j] += h_work[j - peak_lag]
                
                # Add contributions from nearby lags (smooth max approximation)
                for lag in range(max(0, peak_lag - 5), min(N, peak_lag + 6)):
                    if lag == peak_lag:
                        continue
                    w = 0.05 * np.exp(-0.5 * abs(lag - peak_lag))
                    for j in range(N):
                        if 0 <= j + lag < N:
                            grad[j] += w * h_work[j + lag]
                        if 0 <= j - lag < N:
                            grad[j] += w * h_work[j - lag]
                
                # Line search
                current_v = _value_normalized(h_work)
                best_step = 0.0
                best_v_trial = current_v
                for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    h_trial = h_work - step * grad
                    v_trial = _value_normalized(h_trial)
                    if v_trial < best_v_trial:
                        best_v_trial = v_trial
                        best_step = step
                
                if best_step > 0:
                    h_work = h_work - best_step * grad
                    h_candidate = h_work * s_init
                    update_best(h_candidate)
                
                if best_step < 1e-7:
                    break
        except Exception:
            pass
    
    # Phase 5: Coordinate descent on full vector
    if deadline_ok():
        try:
            h_work = best_h.copy()
            s_init = np.sum(h_work)
            if abs(s_init) < 1e-12:
                h_work = np.ones(N)
                s_init = N
            else:
                h_work = h_work / s_init
            
            step_size = 0.02
            rng = np.random.RandomState(123)
            for it in range(300):
                if not deadline_ok():
                    break
                improved = False
                current_v = _value_normalized(h_work)
                for i in range(0, N, 3):  # every 3rd coordinate for speed
                    for delta in [step_size, -step_size]:
                        h_trial = h_work.copy()
                        h_trial[i] += delta
                        v_trial = _value_normalized(h_trial)
                        if v_trial < current_v:
                            h_work = h_trial
                            current_v = v_trial
                            improved = True
                            break
                if not improved:
                    step_size *= 0.7
                    if step_size < 1e-7:
                        break
                update_best(h_work * s_init)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
