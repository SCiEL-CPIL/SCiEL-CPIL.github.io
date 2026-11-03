"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _circular_score(h):
    """
    Compute max|circular autoconv| / (sum h)^2 efficiently.
    Circular autoconv: h ⊛ h = IFFT(|FFT(h)|^2)
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * H).real
    return np.max(np.abs(ac)) / (s * s)

def _freq_objective_circular(x):
    """
    Objective in frequency domain for circular autoconvolution.
    x contains:
    - x[0] = H[0].real (DC, real for real h)
    - x[1] = H[1].real, x[2] = H[1].imag
    - ... for k=1 to N//2
    For real-valued h, H[k] = conj(H[N-k]).
    """
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    
    H0 = H[0]
    if abs(H0) < 1e-12:
        return 1e10
    
    # Circular autoconv = IFFT(|H|^2)
    ac = np.fft.ifft(H * H).real
    return np.max(np.abs(ac)) / (H0 * H0)

def _freq_optimize_circular():
    """
    Optimize in frequency domain for circular autoconvolution.
    Key insight: minimize max_k |IFFT(|H|^2)[k]| / H[0]^2.
    Theoretical minimum is 1/N, achieved when h is constant.
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    # Start with constant h: H[0]=1, H[k]=0 for k>0
    x0 = np.zeros(n_vars)
    x0[0] = 1.0
    
    best_x = x0.copy()
    best_score = _freq_objective_circular(x0)
    
    # Try different initializations
    initializations = [x0.copy()]
    
    # Small perturbations around constant
    rng = np.random.default_rng(42)
    for scale in [0.01, 0.05, 0.1, 0.2]:
        x_init = x0.copy()
        x_init[1:] = rng.normal(0, scale, size=2*n_half)
        initializations.append(x_init)
    
    # Decaying spectrum
    for alpha in [1.0, 2.0, 3.0]:
        x_init = x0.copy()
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = 0.1 / (1.0 + k**alpha)
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    for x_init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _freq_objective_circular,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Convert to time domain
    H = np.zeros(N, dtype=complex)
    H[0] = best_x[0]
    for k in range(1, n_half + 1):
        H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
        H[N - k] = np.conj(H[k])
    
    h = np.fft.ifft(H).real
    return h, best_score

def _score(h):
    """Compute the objective value: max|linear autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline with constant h
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Optimize for circular autoconv (fast proxy), then evaluate with linear
    if time.time() < deadline - 20:
        h_circ, circ_score = _freq_optimize_circular()
        # Evaluate with actual linear autoconv
        lin_score = _score(h_circ)
        if lin_score < best_score:
            best_score = lin_score
            best_h = h_circ.copy()
    
    # Strategy 2: Try to further refine in time domain
    if time.time() < deadline - 5:
        h_ref = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            rng = np.random.default_rng(int(time.time() * 1000) % 10000)
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.005)
                h_ref[idx] += delta
                new_val = _score(h_ref)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_ref[idx] -= delta
            if not improved:
                break
        if best_local < best_score:
            best_score = best_local
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END
