"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective. h is normalized so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return 2.0 * sm / N

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses soft-max surrogate with spectral initialization and coordinate polish.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (score 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Spectral initialization - try to flatten the power spectrum
    # The autoconvolution peak is related to the L4 norm. Minimize sum(h^4) s.t. sum(h)=N
    # This pushes toward uniform, but we can add structured perturbations.
    
    # Try different smooth structured signals
    t = np.arange(N) / N
    candidates = []
    
    # Sine waves with various frequencies and amplitudes
    for m in range(1, 50):
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * t + 0.5)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            h = h * (N / s)
            v = _objective_exact(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Cosine waves
    for m in range(1, 50):
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * t)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            h = h * (N / s)
            v = _objective_exact(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Multi-frequency
    for m1 in range(1, 10):
        for m2 in range(m1 + 1, 15):
            for a1 in [-0.5, 0, 0.5]:
                for a2 in [-0.5, 0, 0.5]:
                    h = 1.0 + a1 * np.sin(2.0 * np.pi * m1 * t) + a2 * np.cos(2.0 * np.pi * m2 * t)
                    s = np.sum(h)
                    if abs(s) < 1e-6:
                        continue
                    h = h * (N / s)
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    if time.time() > deadline - 100:
        return best_h
    
    # Phase 2: DCT subspace optimization with soft-max
    rng = np.random.RandomState(42)
    
    # Generate DCT basis
    K_max = 25
    basis = np.zeros((N, K_max))
    for k in range(K_max):
        for i in range(N):
            basis[i, k] = np.cos(np.pi * (2 * i + 1) * k / (2 * N))
        norm = np.linalg.norm(basis[:, k])
        if norm > 1e-10:
            basis[:, k] /= norm
    
    for K in [5, 10, 15, 20, 25]:
        if time.time() > deadline - 60:
            break
        
        b_basis = basis[:, :K]
        
        def obj_coeff(coeffs):
            b0 = coeffs[0]
            c = coeffs[1:]
            h = b0 + b_basis @ c
            if not np.all(np.isfinite(h)):
                return 1e10
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, 30.0)
            return 2.0 * sm / N
        
        mean_h = np.mean(best_h)
        c_init = np.zeros(K)
        for k in range(K):
            c_init[k] = np.dot(best_h - mean_h, b_basis[:, k])
        coeffs_init = np.concatenate([[mean_h], c_init])
        
        for trial in range(3):
            if time.time() > deadline - 60:
                break
            
            if trial == 0:
                coeffs_try = coeffs_init.copy()
            else:
                coeffs_try = coeffs_init + 0.05 * rng.randn(K + 1)
            
            try:
                result = minimize(
                    obj_coeff,
                    coeffs_try,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                coeffs_opt = result.x
                b_opt = coeffs_opt[0]
                c_opt = coeffs_opt[1:]
                h_opt = b_opt + b_basis @ c_opt
                
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    if time.time() > deadline - 40:
        return best_h
    
    # Phase 3: Full-dimension soft-max optimization
    for beta in [15.0, 30.0, 60.0, 100.0]:
        if time.time() > deadline - 30:
            break
        
        def obj_full(h):
            return _objective_soft(h, beta)
        
        rng2 = np.random.RandomState(100 + int(beta))
        
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.03 * rng2.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            
            try:
                result = minimize(
                    obj_full,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    if time.time() > deadline - 20:
        return best_h
    
    # Phase 4: High-precision polish with higher beta
    for beta in [200.0, 400.0]:
        if time.time() > deadline - 15:
            break
        
        def obj_polish(h):
            return _objective_soft(h, beta)
        
        try:
            result = minimize(
                obj_polish,
                best_h.copy(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            h_opt = result.x
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _objective_exact(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    if time.time() > deadline - 10:
        return best_h
    
    # Phase 5: Coordinate descent polish
    h = best_h.copy()
    step = 0.02
    for iteration in range(60):
        if time.time() > deadline - 5:
            break
        improved = False
        idx = np.random.permutation(N)
        for i in idx:
            if time.time() > deadline - 5:
                break
            old = h[i]
            for delta in [step, -step]:
                h[i] = old + delta
                s = np.sum(h)
                if abs(s) < 1e-10:
                    h[i] = old
                    continue
                v = _objective_exact(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old
        if not improved:
            step *= 0.5
            if step < 0.001:
                break
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
