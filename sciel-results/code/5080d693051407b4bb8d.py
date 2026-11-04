"""
Flat-spectrum phase optimization with epigraph formulation and local polish.

Strategy:
- Fix the magnitude spectrum to be flat (equal energy per frequency bin).
- Optimize the phase vector using a smooth epigraph formulation:
    minimize  z
    s.t.     z >= |autoconv(x)|_k  for all k
  approximated with soft-max / log-sum-exp for differentiability.
- The signal is normalized so sum(x)=1 (scale-invariant metric).
- After gradient-based optimization, do a fast local search polish.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_values(x):
    """Full linear autoconvolution via FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _autoconv_max_abs(x):
    return float(np.max(np.abs(_autoconv_values(x))))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=80.0):
    """Smooth approximation to max(|conv|)."""
    abs_conv = np.abs(conv)
    m = np.max(abs_conv)
    shifted = abs_conv - m
    return m + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=80.0):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_values(x)
    return _soft_max_abs(conv, beta) / (s * s)

def _phase_objective(phi, magnitude, beta=80.0):
    """
    Objective in terms of phase vector phi (length = N//2 + 1).
    magnitude is the fixed |X(k)| vector (length N//2 + 1).
    We reconstruct x from X(k) = magnitude * exp(i*phi).
    """
    # Build complex spectrum
    X = magnitude * np.exp(1j * phi)
    # Ensure real signal: X[k] = conj(X[N-k])
    # For even N, X[N/2] must be real; for odd, all pairs.
    # We enforce Hermitian symmetry explicitly.
    n_fft = N
    X_full = np.zeros(n_fft, dtype=complex)
    X_full[:len(magnitude)] = X
    # Enforce Hermitian symmetry
    for k in range(1, n_fft // 2 + 1):
        idx = n_fft - k
        if idx < len(magnitude):
            X_full[idx] = np.conj(X_full[k])
        else:
            X_full[idx] = np.conj(X_full[k])
    # If N is even, index N/2 is real
    if n_fft % 2 == 0:
        X_full[n_fft // 2] = np.real(X_full[n_fft // 2])
    
    x = np.fft.ifft(X_full).real[:N]
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_values(x)
    return _soft_max_abs(conv, beta) / (s * s)

def _epigraph_objective(params, magnitude, beta=80.0):
    """
    Epigraph formulation: params = [phi (len M), z (scalar)]
    M = len(magnitude) = N//2 + 1
    """
    M = len(magnitude)
    phi = params[:M]
    z = params[M]
    
    X = magnitude * np.exp(1j * phi)
    n_fft = N
    X_full = np.zeros(n_fft, dtype=complex)
    X_full[:len(magnitude)] = X
    for k in range(1, n_fft // 2 + 1):
        idx = n_fft - k
        if idx < len(magnitude):
            X_full[idx] = np.conj(X_full[k])
        else:
            X_full[idx] = np.conj(X_full[k])
    if n_fft % 2 == 0:
        X_full[n_fft // 2] = np.real(X_full[n_fft // 2])
    
    x = np.fft.ifft(X_full).real[:N]
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_values(x)
    abs_conv = np.abs(conv)
    
    # Soft max
    smax = _soft_max_abs(conv, beta)
    
    # Epigraph penalty: encourage z >= smax
    # Use a smooth hinge: max(0, smax - z)^2 / scale + z
    penalty = np.maximum(0.0, smax - z) ** 2 / (1e-4 + z)
    return z + penalty

def _spectral_design_optimize(magnitude, deadline, rng, M_params=None):
    """
    Optimize phase vector using L-BFGS-B.
    Returns (x, metric_value).
    """
    M = len(magnitude)
    if M_params is None:
        M_params = M
    
    # Initial random phases
    phi_init = rng.uniform(-np.pi, np.pi, size=M)
    
    best_phi = phi_init.copy()
    best_val = _phase_objective(best_phi, magnitude, 60.0)
    
    for beta in [40.0, 60.0, 80.0, 100.0]:
        if time.time() > deadline - 8:
            break
        
        # Convert to epigraph params
        M = len(magnitude)
        # Initial z: estimate from current
        X = magnitude * np.exp(1j * best_phi)
        n_fft = N
        X_full = np.zeros(n_fft, dtype=complex)
        X_full[:len(magnitude)] = X
        for k in range(1, n_fft // 2 + 1):
            idx = n_fft - k
            if idx < len(magnitude):
                X_full[idx] = np.conj(X_full[k])
            else:
                X_full[idx] = np.conj(X_full[k])
        if n_fft % 2 == 0:
            X_full[n_fft // 2] = np.real(X_full[n_fft // 2])
        x_est = np.fft.ifft(X_full).real[:N]
        s_est = np.sum(x_est)
        if abs(s_est) < 1e-12:
            continue
        conv_est = _autoconv_values(x_est)
        z_init = float(np.max(np.abs(conv_est))) / (s_est * s_est) * 1.2
        
        params_init = np.concatenate([best_phi, [z_init]])
        
        res = minimize(
            _epigraph_objective,
            params_init,
            args=(magnitude, beta),
            method='L-BFGS-B',
            options={'maxiter': 80, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-4}
        )
        
        phi_new = res.x[:M]
        val_new = _phase_objective(phi_new, magnitude, beta)
        
        if val_new < best_val:
            best_val = val_new
            best_phi = phi_new
    
    # Reconstruct final x
    X = magnitude * np.exp(1j * best_phi)
    n_fft = N
    X_full = np.zeros(n_fft, dtype=complex)
    X_full[:len(magnitude)] = X
    for k in range(1, n_fft // 2 + 1):
        idx = n_fft - k
        if idx < len(magnitude):
            X_full[idx] = np.conj(X_full[k])
        else:
            X_full[idx] = np.conj(X_full[k])
    if n_fft % 2 == 0:
        X_full[n_fft // 2] = np.real(X_full[n_fft // 2])
    
    x = np.fft.ifft(X_full).real[:N]
    s = np.sum(x)
    if abs(s) < 1e-12:
        return None, float('inf')
    x = x / s
    val = _metric(x)
    return x, val

def _try_local_search_fast(x, deadline, rng, max_rounds=3000):
    """
    Fast local search: perturb random subsets of x and accept if metric improves.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.05
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Perturb a random subset
        k = int(rng.integers(1, min(5, N // 20 + 1)))
        indices = rng.choice(N, size=k, replace=False)
        
        improved = False
        for trial in range(3):
            x_test = best_x.copy()
            x_test[indices] += rng.normal(0, step_scale, size=k)
            s = np.sum(x_test)
            if abs(s) < 1e-12:
                continue
            x_test = x_test / s
            v = _metric(x_test)
            if v < best_val:
                best_x = x_test
                best_val = v
                no_improve = 0
                improved = True
                break
        
        if not improved:
            no_improve += 1
    
    return best_x, best_val

def _final_polish(x, deadline, rng):
    """
    Final polish: coordinate descent on individual elements.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            orig = best_x[i]
            # Try a few perturbations
            for delta in [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]:
                x_test = best_x.copy()
                x_test[i] = orig + delta
                s = np.sum(x_test)
                if abs(s) < 1e-12:
                    continue
                x_test = x_test / s
                v = _metric(x_test)
                if v < best_val:
                    best_x = x_test
                    best_val = v
                    break
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Initial signal: all ones
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Build flat magnitude spectrum
    # For N=400 (even), rfft gives N//2+1 = 201 coefficients
    n_half = N // 2 + 1
    # Flat magnitude: constant, but we can taper slightly to reduce edge effects
    # Use a gentle cosine taper
    k = np.arange(n_half)
    # Slight taper to reduce Gibbs-like artifacts at boundaries
    taper = 0.5 + 0.5 * np.cos(np.pi * k / (n_half - 1))
    taper[0] = 1.0  # Keep DC
    magnitude = np.ones(n_half) * taper
    
    # Try multiple random phase initializations
    for restart in range(5):
        if time.time() > deadline - 25:
            break
        
        # Slightly vary the taper per restart
        if restart > 0:
            taper_mod = 0.5 + 0.5 * np.cos(np.pi * k / (n_half - 1) * (1 + 0.1 * restart))
            taper_mod[0] = 1.0
            magnitude = np.ones(n_half) * taper_mod
        else:
            magnitude = np.ones(n_half) * taper
        
        try:
            x_opt, val_opt = _spectral_design_optimize(magnitude, deadline - 15, rng)
            if x_opt is not None and val_opt < best_val:
                best_x = x_opt
                best_val = val_opt
        except Exception:
            continue
    
    # Local search polish
    if time.time() < deadline - 10:
        x_ls, val_ls = _try_local_search_fast(best_x, deadline - 5, rng)
        if val_ls < best_val:
            best_x = x_ls
            best_val = val_ls
    
    # Final coordinate polish
    if time.time() < deadline - 2:
        x_fp, val_fp = _final_polish(best_x, deadline - 0.5, rng)
        if val_fp < best_val:
            best_x = x_fp
            best_val = val_fp
    
    return best_x
