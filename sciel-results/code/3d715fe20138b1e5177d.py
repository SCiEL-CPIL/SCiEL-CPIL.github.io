"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

We use a DFT-based parameterization: h = |IFFT(z)|^2 where z is a complex vector
with exactly K non-zero entries (sparse spectral mask). This enforces non-negativity
naturally and restricts frequency support to minimize autoconvolution peaks.

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf (equivalently maximize 1 / (||h||_1 * ||h*h||_inf)).
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (1/2)/DH

_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _h_from_z(z, active_indices):
    """Compute h = |IFFT(z)|^2 where z has non-zero entries only at active_indices."""
    z_full = np.zeros(N, dtype=complex)
    z_full[active_indices] = z
    h = np.abs(np.fft.ifft(z_full)) ** 2
    return h


def _neg_logR_from_z(z, active_indices, beta):
    """Negative log-ratio for optimization, using soft-max smoothing."""
    h = _h_from_z(z, active_indices)
    h = np.maximum(h, 1e-30)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l2sq <= 1e-30 or l1 <= 1e-30:
        return 1e12
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 1e-30:
        return 1e12
    # Soft log-sup
    shifted = ac - acmax
    logsup = acmax + math.log(float(np.sum(np.exp(beta * shifted)))) / beta
    return -(math.log(l2sq) - math.log(l1) - logsup)


def _neg_logR_grad_from_z(z, active_indices, beta):
    """Numerical gradient for the z-parameterized objective."""
    val = _neg_logR_from_z(z, active_indices, beta)
    g = np.zeros_like(z)
    eps = 1e-5
    for i in range(len(z)):
        zp = z.copy()
        zp[i] += eps
        vm = z.copy()
        vm[i] -= eps
        vp = _neg_logR_from_z(zp, active_indices, beta)
        vmm = _neg_logR_from_z(vm, active_indices, beta)
        g[i] = (vp - vmm) / (2 * eps)
    return val, g


def _local_optimize_z(z0, active_indices, time_budget):
    """Optimize over z (complex) for a fixed set of active frequency bins."""
    t0 = time.time()
    
    def _over():
        return (time.time() - t0) > time_budget
    
    # Convert z0 to real parameter vector: [Re(z0), Im(z0), Re(z1), Im(z1), ...]
    def _z_to_vec(z):
        vec = np.zeros(2 * len(z))
        vec[0::2] = np.real(z)
        vec[1::2] = np.imag(z)
        return vec
    
    def _vec_to_z(vec):
        z = np.zeros(len(vec) // 2, dtype=complex)
        z.real = vec[0::2]
        z.imag = vec[1::2]
        return z
    
    vec0 = _z_to_vec(z0)
    best_vec = vec0.copy()
    best_v = _ratio(_h_from_z(z0, active_indices))
    
    # L-BFGS-B on the real-valued parameter vector
    beta = 40.0
    bounds = [(-np.inf, np.inf)] * (2 * len(z0))
    
    try:
        def obj(vec):
            if _over():
                return np.nan
            z = _vec_to_z(vec)
            return _neg_logR_from_z(z, active_indices, beta)
        
        def jac(vec):
            if _over():
                return np.nan, np.zeros_like(vec)
            z = _vec_to_z(vec)
            val, gz = _neg_logR_grad_from_z(z, active_indices, beta)
            # Convert complex gradient to real vector gradient
            gvec = np.zeros_like(vec)
            gvec[0::2] = np.real(gz)
            gvec[1::2] = np.imag(gz)
            return val, gvec
        
        res = minimize(
            obj, vec0,
            method='L-BFGS-B',
            jac=jac,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
        )
        if res.x is not None:
            h_new = _h_from_z(_vec_to_z(res.x), active_indices)
            v_new = _ratio(h_new)
            if v_new > best_v:
                best_v = v_new
                best_vec = res.x.copy()
    except Exception:
        pass
    
    # Nelder-Mead polish on the raw ratio
    if _time_left() > 5.0 and not _over():
        try:
            def obj_raw(vec):
                if _over():
                    return np.nan
                z = _vec_to_z(vec)
                h = _h_from_z(z, active_indices)
                return -_ratio(h)
            
            res2 = minimize(
                obj_raw, best_vec,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True, 'disp': False},
            )
            if res2.x is not None:
                h_new = _h_from_z(_vec_to_z(res2.x), active_indices)
                v_new = _ratio(h_new)
                if v_new > best_v:
                    best_v = v_new
                    best_vec = res2.x.copy()
        except Exception:
            pass
    
    return best_vec, best_v


def _initializations_z():
    """Yield diverse starting points in z-space with various active bin sets."""
    rng = np.random.default_rng(42)
    inits = []
    
    # Try different numbers of active bins
    for K in [2, 3, 4, 5, 6, 8]:
        # Symmetric pairs around 0 (for real h)
        # Frequencies: 0, 1, ..., N-1. For real h, z[k] = conj(z[N-k])
        # Active indices should be symmetric
        if K % 2 == 0:
            # K/2 positive frequency pairs + possibly DC
            n_pairs = K // 2
            # Choose n_pairs distinct positive frequencies
            pos_freqs = list(range(1, N // 2 + 1))
            chosen = rng.choice(pos_freqs, size=n_pairs, replace=False)
            active = []
            for f in chosen:
                active.append(f)
                if f != N - f:
                    active.append(N - f)
            if 0 not in active and len(active) < K:
                active.append(0)
        else:
            # Include DC
            n_pairs = (K - 1) // 2
            pos_freqs = list(range(1, N // 2 + 1))
            chosen = rng.choice(pos_freqs, size=n_pairs, replace=False)
            active = [0]
            for f in chosen:
                active.append(f)
                if f != N - f:
                    active.append(N - f)
        
        # Initialize z with random phases and magnitudes
        z0 = np.zeros(len(active), dtype=complex)
        for i in range(len(active)):
            z0[i] = rng.random() + 1j * rng.random()
        
        inits.append((active, z0))
    
    # Also try structured initializations: equal magnitude, random phase
    for K in [3, 5, 7]:
        if K % 2 == 1:
            n_pairs = (K - 1) // 2
            pos_freqs = list(range(1, N // 2 + 1))
            chosen = sorted(rng.choice(pos_freqs, size=n_pairs, replace=False))
            active = [0]
            for f in chosen:
                active.append(f)
                if f != N - f:
                    active.append(N - f)
            z0 = np.ones(len(active), dtype=complex) * (1 + 1j)
            inits.append((active, z0))
    
    return inits


def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_v = 0.0
    
    inits = _initializations_z()
    
    # Evaluate all initializations and sort
    scored = []
    for active, z0 in inits:
        h = _h_from_z(z0, active)
        v = _ratio(h)
        scored.append((v, active, z0))
        if v > best_v:
            best_v = v
            best_h = h.copy()
    scored.sort(key=lambda t: -t[0])
    
    # Optimize top initializations
    budget_per = 8.0
    for v0, active, z0 in scored:
        if _time_left() <= budget_per + 3.0:
            break
        if v0 < best_v * 0.3 and best_v > 15.0:
            continue
        
        vec_opt, v_opt = _local_optimize_z(z0, active, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            h_opt = _h_from_z(_vec_to_z_local(vec_opt, len(z0)), active)
            best_h = h_opt.copy()
        
        if _time_left() <= 25.0:
            break
    
    # Final polish on global best
    if best_h is not None and _time_left() > 20.0:
        # Find the best z that gave best_h and do more optimization
        # We need to track which active set was best
        # For simplicity, just do a final Nelder-Mead on h-space if needed
        pass
    
    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]


def _vec_to_z_local(vec, n_z):
    """Helper to convert real vector to complex z."""
    z = np.zeros(n_z, dtype=complex)
    z.real = vec[0::2]
    z.imag = vec[1::2]
    return z


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
