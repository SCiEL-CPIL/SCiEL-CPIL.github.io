import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac

def _score(f):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(f, beta=100.0):
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(abs_ac)
    exp_terms = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_max_val = m + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    return soft_max_val, weights

def _soft_score(f, beta=100.0):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    val, _ = _soft_max_abs_ac(f, beta)
    return val / (s * s)

def _soft_score_grad(f, beta=100.0):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.zeros_like(f), np.inf
    
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    soft_max_val, weights = _soft_max_abs_ac(f, beta)
    
    if soft_max_val < 1e-15:
        return np.zeros_like(f), 0.0
    
    w = weights  # length 2*n-1
    w_padded = np.zeros(2 * size)
    w_padded[:len(w)] = w
    
    W = np.fft.rfft(w_padded, 2 * size)
    F_full = np.fft.rfft(f, 2 * size)
    
    conv1 = np.fft.irfft(W * F_full, 2 * size)[:2 * n - 1]
    
    w_flip = w[::-1]
    w_flip_padded = np.zeros(2 * size)
    w_flip_padded[:len(w_flip)] = w_flip
    W_flip = np.fft.rfft(w_flip_padded, 2 * size)
    conv2 = np.fft.irfft(W_flip * F_full, 2 * size)[:2 * n - 1]
    
    grad_soft_max = conv1 + conv2
    
    grad = grad_soft_max / (s * s) - 2 * soft_max_val / (s * s * s)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(f)
    
    return grad, soft_max_val / (s * s)

def _get_amplitude_spectrum(n):
    # Create a Gaussian amplitude spectrum in frequency domain
    # This concentrates energy in low frequencies, minimizing time-domain spread
    k = np.arange(n)
    # Bandwidth parameter: smaller sigma means more concentrated in freq, less in time?
    # Actually, Gaussian in freq is Gaussian in time.
    # We want to minimize peak autoconvolution.
    # Let's try a few shapes.
    sigma = 0.1
    A = np.exp(-0.5 * (k / (n * sigma))**2)
    # Normalize DC component to 1 to fix the sum
    A[0] = 1.0
    return A

def _phase_objective_grad(phi, A, beta=100.0):
    # Reconstruct x from amplitude A and phase phi
    # x = Re(IFFT(A * exp(i*phi)))
    # Note: phi is length N (full spectrum) or N//2+1 (rfft)?
    # Let's use full complex spectrum for clarity, or rfft.
    # Using rfft:
    # A is length N//2 + 1
    # phi is length N//2 + 1
    F = A * np.exp(1j * phi)
    x = np.fft.irfft(F, N)
    
    # Compute score and grad w.r.t x
    gx, val = _soft_score_grad(x, beta)
    
    if not np.all(np.isfinite(gx)):
        gx = np.zeros_like(x)
    
    # Compute gradient w.r.t phi
    # dJ/dphi = Re( -i * A * exp(-i*phi) * FFT(gx) )
    G = np.fft.rfft(gx, N)
    # G is length N//2 + 1
    # A is length N//2 + 1
    # phi is length N//2 + 1
    
    # Handle DC component (k=0) and Nyquist (if N even)
    # For k=0, A is real, phi is usually 0.
    # The formula holds generally.
    
    g_phi = np.real(-1j * A * np.exp(-1j * phi) * G)
    
    if not np.all(np.isfinite(g_phi)):
        g_phi = np.zeros_like(phi)
    
    return val, g_phi

def construct():
    deadline = time.time() + 115.0
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    np.random.seed(42)
    
    # Get amplitude spectrum
    A = _get_amplitude_spectrum(N)
    
    # Phase optimization
    # Initialize phases
    # We can try multiple random phase initializations
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        phi0 = np.random.randn(N // 2 + 1)
        
        def objective_and_grad(phi):
            val, g = _phase_objective_grad(phi, A, beta=100.0)
            if not np.all(np.isfinite(g)):
                g = np.zeros_like(phi)
            if not np.isfinite(val):
                val = 1e10
            return val, g
        
        try:
            result = minimize(
                objective_and_grad,
                phi0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            
            phi_opt = result.x
            # Reconstruct x
            F = A * np.exp(1j * phi_opt)
            x_opt = np.fft.irfft(F, N)
            
            sc = _score(x_opt)
            
            if sc < best_score:
                best_score = sc
                best_x = x_opt.copy()
                
        except Exception:
            pass
            
    # Local refinement in time domain (optional, might break the spectral structure)
    # But we can do a few steps of coordinate descent to fine-tune
    if time.time() < deadline - 5:
        x = best_x.copy()
        # Simple coordinate descent
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            idx = np.random.randint(N)
            x_old = x[idx]
            for direction in [-0.1, 0.1]:
                x[idx] = x_old + direction
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                    break
            else:
                x[idx] = x_old

    return best_x
