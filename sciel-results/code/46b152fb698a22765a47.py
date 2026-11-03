import numpy as np
import time
from scipy.optimize import minimize_scalar

def _autoconv(f):
    """Compute autoconvolution c_k = sum_i f_i * f_{i+k} for k in [-(n-1), n-1].
    Returns array of length 2n-1 where index n-1 corresponds to k=0."""
    n = len(f)
    # Use FFT: zero-pad to 2n, compute linear autoconvolution
    F = np.fft.fft(f, n=2*n)
    ac_full = np.fft.ifft(F * np.conj(F))[:n].real
    # ac_full[k] = sum_{j=0}^{n-1-k} f[j] * f[j+k] for k=0,...,n-1
    full_ac = np.zeros(2*n-1)
    full_ac[n-1:] = ac_full
    full_ac[:n-1] = ac_full[1:][::-1]
    return full_ac

def _compute_score(f):
    """Compute the normalized max of autoconvolution.
    Score = max_k c_k / c_0 where c_0 = sum f_i^2."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    c0 = np.dot(f, f)
    if c0 == 0:
        return 0.0
    ac = _autoconv(f)
    max_ac = np.max(ac)
    return max_ac / c0

def _spectral_param_to_f(params, n=600):
    """Convert spectral parameters to a nonnegative step function.
    
    params: complex array of length n (DFT coefficients)
    We enforce non-negativity by projecting the inverse DFT to be nonnegative.
    """
    # params is the DFT of f
    f = np.fft.ifft(params).real
    # Project to nonnegative
    f = np.maximum(f, 0)
    return f

def _f_to_spectral(f, n=600):
    """Convert time-domain f to spectral domain (DFT)."""
    return np.fft.fft(f)

def _spectral_objective(params, n=600):
    """Objective function in spectral domain: minimize max autoconv / c0.
    
    params: complex array of DFT coefficients
    """
    f = _spectral_param_to_f(params, n)
    if np.all(f == 0):
        return 0.0
    return _compute_score(f)

def _spectral_coordinate_descent(params, n=600, max_iter=200, tol=1e-10):
    """Coordinate descent in spectral domain.
    
    For each frequency bin, optimize the real and imaginary parts of the DFT coefficient
    while keeping others fixed, using minimize_scalar.
    """
    params = np.array(params, dtype=np.complex128, copy=True)
    best_score = _spectral_objective(params, n)
    
    for it in range(max_iter):
        improved = False
        for k in range(n):
            # Optimize real part of params[k]
            def obj_real(x):
                old = params[k]
                params[k] = x + 1j * old.imag
                s = _spectral_objective(params, n)
                return s
            
            res = minimize_scalar(obj_real, bounds=(-2.0 * np.abs(params[k]), 2.0 * np.abs(params[k])),
                                  method='bounded', options={'xatol': 1e-8})
            if res.fun < best_score - tol:
                old = params[k]
                params[k] = res.x + 1j * old.imag
                best_score = res.fun
                improved = True
            
            # Optimize imaginary part of params[k]
            def obj_imag(x):
                old = params[k]
                params[k] = old.real + 1j * x
                s = _spectral_objective(params, n)
                return s
            
            res = minimize_scalar(obj_imag, bounds=(-2.0 * np.abs(params[k]), 2.0 * np.abs(params[k])),
                                  method='bounded', options={'xatol': 1e-8})
            if res.fun < best_score - tol:
                old = params[k]
                params[k] = old.real + 1j * res.x
                best_score = res.fun
                improved = True
        
        if not improved:
            break
    
    return params, best_score

def _time_coordinate_descent(f, max_iter=50, tol=1e-10):
    """Coordinate descent in time domain to polish the solution."""
    n = len(f)
    f = np.array(f, dtype=np.float64, copy=True)
    best_score = _compute_score(f)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            old_val = f[i]
            
            # Optimize f[i] using minimize_scalar
            def obj(x):
                f[i] = max(x, 0)
                return _compute_score(f)
            
            # Bound: try reasonable range
            if old_val > 1e-12:
                lo = 0.0
                hi = 2.0 * old_val + 1.0
            else:
                lo = 0.0
                hi = 1.0
            
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.fun < best_score - tol:
                f[i] = max(res.x, 0)
                best_score = res.fun
                improved = True
            else:
                f[i] = old_val
        
        if not improved:
            break
    
    return f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 600
    
    # Initialize with uniform
    f_init = np.ones(n, dtype=np.float64)
    best_f = f_init.copy()
    best_score = _compute_score(f_init)
    
    # Phase 1: Spectral coordinate descent
    if time.time() < deadline - 30.0:
        params_init = _f_to_spectral(f_init, n)
        params, score_spec = _spectral_coordinate_descent(params_init, n, max_iter=100)
        f_spec = _spectral_param_to_f(params, n)
        score_spec = _compute_score(f_spec)
        if score_spec < best_score:
            best_score = score_spec
            best_f = f_spec
    
    # Phase 2: Time-domain coordinate descent polish
    if time.time() < deadline - 10.0:
        f_cd, score_cd = _time_coordinate_descent(best_f, max_iter=30)
        if score_cd < best_score:
            best_score = score_cd
            best_f = f_cd
    
    # Phase 3: More spectral refinement if time allows
    if time.time() < deadline - 5.0:
        params = _f_to_spectral(best_f, n)
        params, score_spec2 = _spectral_coordinate_descent(params, n, max_iter=50)
        f_spec2 = _spectral_param_to_f(params, n)
        score_spec2 = _compute_score(f_spec2)
        if score_spec2 < best_score:
            best_score = score_spec2
            best_f = f_spec2
    
    return list(best_f)
