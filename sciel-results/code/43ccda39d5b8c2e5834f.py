"""Spectral phase optimization for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max|ac| using log-sum-exp.
    soft_max(|ac|) = (1/beta) * log(sum(exp(beta*|ac|)))
    This is differentiable and approximates max|ac| as beta -> inf.
    """
    abs_ac = np.abs(ac)
    # Shift for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    exp_vals = np.exp(beta * (abs_ac - max_val))
    log_sum = np.log(np.sum(exp_vals))
    return (max_val + log_sum / beta)

def _objective_smooth(x, beta=50.0):
    """
    Smooth objective: soft_max(|autoconv|) / (sum x)^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    smax = _soft_max_abs(ac, beta)
    return smax / (s * s)

def _objective_grad(x, beta=50.0):
    """
    Compute gradient of smooth objective using finite differences or analytical approach.
    For efficiency, use a central difference approximation on a subset of coordinates.
    """
    grad = np.zeros(N)
    eps = 1e-4
    f0 = _objective_smooth(x, beta)
    
    for i in range(0, N, 2):  # Sample every other coordinate for speed
        if _check_deadline():
            return grad
        
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective_smooth(x_plus, beta)
        
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = _objective_smooth(x_minus, beta)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _flat_top_magnitude(N, alpha=0.54):
    """
    Generate a flat-top (Hamming-like) magnitude spectrum.
    This gives low sidelobes in the autoconvolution domain.
    """
    k = np.arange(N)
    kf = k / N
    
    # Hamming window shape: 0.54 - 0.46*cos(2*pi*kf)
    # But we want it centered and symmetric
    mag = 0.54 - 0.46 * np.cos(2 * np.pi * kf)
    
    # Ensure non-negative and symmetric
    mag = np.maximum(mag, 0.0)
    
    # Make it symmetric for real signal
    # For DFT of real signal: X[N-k] = conj(X[k])
    # So magnitude should be symmetric: mag[N-k] = mag[k]
    # Our formula already gives this since cos(2*pi*(N-k)/N) = cos(2*pi - 2*pi*k/N) = cos(2*pi*k/N)
    
    # Normalize so that sum(x) is reasonable
    # X[0] = sum(x), so mag[0] controls the integral
    # Scale so that mag[0] = 1 (unit integral)
    if mag[0] > 1e-10:
        mag = mag / mag[0]
    
    return mag

def _phase_gradient_descent(mag, initial_phase, num_iters=5000, lr=0.01, beta=50.0):
    """
    Optimize phase to minimize the smooth objective.
    x = ifft(mag * exp(i*phase))
    We optimize phase directly.
    """
    phase = initial_phase.copy()
    best_val = 1e10
    best_x = None
    
    # Precompute for efficiency
    L_fft = 1
    while L_fft < 2*N-1:
        L_fft *= 2
    
    for it in range(num_iters):
        if _check_deadline():
            break
        
        # Compute x from current phase
        X = mag * np.exp(1j * phase)
        x = np.fft.ifft(X).real
        
        # Compute objective
        val = _objective_smooth(x, beta)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Compute gradient with respect to phase
        # d/dphase_j [soft_max(|ac|) / (sum x)^2]
        # Use finite differences on phase
        grad_phase = np.zeros(N)
        eps = 1e-3
        
        # Sample a subset of phase coordinates for speed
        sample_indices = np.random.default_rng(it % 1000).choice(N, size=min(20, N), replace=False)
        
        for idx in sample_indices:
            phase_plus = phase.copy()
            phase_plus[idx] += eps
            
            X_plus = mag * np.exp(1j * phase_plus)
            x_plus = np.fft.ifft(X_plus).real
            f_plus = _objective_smooth(x_plus, beta)
            
            phase_minus = phase.copy()
            phase_minus[idx] -= eps
            
            X_minus = mag * np.exp(1j * phase_minus)
            x_minus = np.fft.ifft(X_minus).real
            f_minus = _objective_smooth(x_minus, beta)
            
            grad_phase[idx] = (f_plus - f_minus) / (2 * eps)
        
        # Update phase
        phase -= lr * grad_phase
        
        # Normalize phase (remove DC component for stability)
        phase -= np.mean(phase)
    
    if best_x is not None:
        return best_x, best_val
    else:
        x = np.fft.ifft(mag * np.exp(1j * phase)).real
        return x, _objective_smooth(x, beta)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Try different flat-top magnitude shapes
    alpha_values = [0.54, 0.5, 0.46, 0.6, 0.55, 0.45, 0.52, 0.48]
    
    for alpha in alpha_values:
        if _check_deadline():
            break
        
        mag = _flat_top_magnitude(N, alpha)
        
        # Try different initial phases
        rng = np.random.default_rng(42)
        initial_phases = [
            np.zeros(N),
            rng.uniform(0, 2*np.pi, N),
            2*np.pi * np.linspace(0, 1, N, endpoint=False),
            2*np.pi * np.linspace(0, 2, N, endpoint=False),
            rng.normal(0, 1, N),
        ]
        
        for init_phase in initial_phases:
            if _check_deadline():
                break
            
            try:
                x_opt, val_opt = _phase_gradient_descent(
                    mag, init_phase, 
                    num_iters=2000, lr=0.005, beta=50.0
                )
                
                # Also evaluate the true (non-smooth) objective
                true_val = _objective(x_opt)
                if true_val < best_val:
                    best_val = true_val
                    best_x = x_opt.copy()
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Local refinement on the best solution found
    try:
        # Use BFGS on the smooth objective
        from scipy.optimize import minimize
        
        def _smooth_obj(x):
            return _objective_smooth(x, beta=100.0)
        
        result = minimize(_smooth_obj, best_x, method='BFGS',
                         options={'maxiter': 200, 'gtol': 1e-5})
        if result.fun < 1e9:
            true_val = _objective(result.x)
            if true_val < best_val:
                best_val = true_val
                best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Final polish with random perturbations
    rng = np.random.default_rng(123)
    step_size = 0.02
    for _ in range(50000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.999)
        else:
            step_size = min(0.1, step_size * 1.01)
    
    return best_x
# EVOLVE-BLOCK-END
