"""Spectral-envelope optimization for signed step functions minimizing max|autoconv|/(sum x)^2.

Structural change: Instead of optimizing x directly with soft-max surrogates, we work in the
spectral domain. The autoconvolution in frequency is |X(f)|^2, and the peak constraint becomes
a constraint on the time-domain signal whose spectrum is |X(f)|^2. We parameterize X(f) as a
smooth spectral envelope and optimize its parameters to minimize the H-infinity-like norm
of the autoconvolution relative to the squared integral.

Key insight: If we write x as the IFFT of X, then autoconv = IFFT(|X|^2). The max|autoconv|
is the peak of a signal whose spectrum is |X|^2. By the uncertainty principle, spreading
energy across frequencies reduces peak. We optimize a parametric family of |X|^2 that
minimizes the L-infinity norm of its IFFT, subject to the area constraint (sum x = 1).

We combine this with the refinement machinery (coordinate descent, peak-targeted polish,
random perturbation) from the current program to get both a good starting point and fine
tuning.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

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

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation using minimal FFT sizes
    g = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[n-j] = cross-correlation
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] = convolution
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_envelope_objective(params):
    """
    Optimize spectral envelope parameters.
    params = [a, tilt, mod, phase_shift]
    where a controls bandwidth, tilt is frequency tilt, mod is modulation, 
    phase_shift is a phase rotation to allow signed sequences.
    
    The spectral magnitude is |X(f)| = exp(-a*f^2) * exp(tilt*f) * (1 + mod*sin(2*pi*f))
    We then take IFFT to get x, normalize to unit sum, and evaluate the objective.
    """
    a, tilt, mod, phase = params[:4]
    
    # Clip parameters to reasonable ranges
    a = np.clip(a, 0.3, 5.0)
    tilt = np.clip(tilt, -1.5, 1.5)
    mod = np.clip(mod, 0.0, 1.0)
    phase = np.clip(phase, -np.pi, np.pi)
    
    # Frequency grid for the spectral envelope
    k = np.arange(N)
    kf = k / N  # Normalized frequency [0, 1)
    
    # Spectral magnitude
    mag = np.exp(-a * kf**2)
    mag = mag * np.exp(tilt * kf)
    mag = mag * (1.0 + mod * 0.5 * np.sin(2 * np.pi * 3 * kf))
    
    # Add phase to allow signed sequences
    phase_vec = phase * kf
    X = mag * np.exp(1j * phase_vec)
    
    # Convert to time domain
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    # Normalize to unit sum
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    
    x = x / s
    
    return _objective(x)

def _spectral_envelope_grad(params):
    """
    Gradient of the spectral envelope objective using finite differences.
    This is slower but more robust for the parametric optimization.
    """
    params = np.array(params)
    grad = np.zeros_like(params)
    eps = 1e-6
    
    f0 = _spectral_envelope_objective(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        
        f_plus = _spectral_envelope_objective(params_plus)
        f_minus = _spectral_envelope_objective(params_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return f0, grad

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try small adjustments to x[j]
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _peak_targeted_polish(x, max_iters=3000):
    """
    Peak-targeted polish: identify dominant autoconvolution peaks and try to reduce them.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac_current = _autoconv_fft(best_x)
        s_current = np.sum(best_x)
        abs_ac_current = np.abs(ac_current)
        max_ac_val = np.max(abs_ac_current)
        
        # Find top K peaks
        K = min(20, 2*N-1)
        peak_indices = np.argsort(abs_ac_current)[-K:]
        
        improved = False
        for peak_idx in peak_indices:
            if _check_deadline():
                break
            
            # The autoconvolution at lag peak_idx is sum_j x[j]*x[peak_idx-j]
            # To reduce it, we can try adjusting magnitudes of x[j] where 
            # x[j]*x[peak_idx-j] contributes positively to the peak
            
            # Get contributing pairs
            j_range = np.arange(max(0, peak_idx - N + 1), min(N, peak_idx + 1))
            if len(j_range) == 0:
                continue
            
            partner_indices = peak_idx - j_range
            valid = (partner_indices >= 0) & (partner_indices < N)
            j_range = j_range[valid]
            
            if len(j_range) == 0:
                continue
            
            # Try small adjustments to reduce the peak
            for j in j_range[:min(50, len(j_range))]:
                if _check_deadline():
                    break
                
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    trial = best_x.copy()
                    trial[j] += delta
                    val = _objective(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(20):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral envelope optimization (STRUCTURAL CHANGE)
    # Try multiple initial points for the spectral envelope parameters
    spectral_initials = [
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.5, 0.5, 0.0, 0.0],
        [1.5, -0.5, 0.0, 0.0],
        [1.0, 0.0, 0.5, 0.0],
        [2.0, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, -0.5],
        [1.5, 0.5, 0.3, 0.3],
        [1.5, -0.5, 0.3, -0.3],
        [2.0, 0.5, 0.5, 0.5],
        [2.0, -0.5, 0.5, -0.5],
    ]
    
    for p0 in spectral_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_spectral_envelope_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                # Extract the optimal x from the spectral parameters
                a, tilt, mod, phase = result.x[:4]
                a = np.clip(a, 0.3, 5.0)
                tilt = np.clip(tilt, -1.5, 1.5)
                mod = np.clip(mod, 0.0, 1.0)
                phase = np.clip(phase, -np.pi, np.pi)
                
                k = np.arange(N)
                kf = k / N
                mag = np.exp(-a * kf**2)
                mag = mag * np.exp(tilt * kf)
                mag = mag * (1.0 + mod * 0.5 * np.sin(2 * np.pi * 3 * kf))
                phase_vec = phase * kf
                X = mag * np.exp(1j * phase_vec)
                x_spec = np.fft.ifft(X).real
                x_spec = x_spec / np.max(np.abs(x_spec))
                s_spec = np.sum(x_spec)
                if abs(s_spec) > 1e-10:
                    x_spec = x_spec / s_spec
                    val_spec = _objective(x_spec)
                    if val_spec < best_val:
                        best_val = val_spec
                        best_x = x_spec.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: L-BFGS-B with smooth surrogate on the best x found so far
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Try multiple beta values for soft-max
    for beta in [10.0, 20.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=3000)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 50000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        # Use smooth surrogate for fast evaluation
        val_smooth, _ = _objective_grad_smooth(x_trial, beta=50.0)
        
        if val_smooth < best_val * 0.999:  # Require clear improvement in surrogate
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 5000 == 0:
            step_size = min(0.5, step_size * 1.05)
            if stall_count % 20000 == 0:
                big_pert = rng.normal(0, step_size * 5, size=N)
                x_big = best_x + big_pert
                val_big = _objective(x_big)
                if val_big < best_val:
                    best_val = val_big
                    best_x = x_big.copy()
                    stall_count = 0
    
    # Phase 8: Final L-BFGS polish with high beta
    if not _check_deadline():
        try:
            def _high_beta_obj(x):
                val, grad = _objective_grad_smooth(x, beta=100.0)
                return val, grad
            
            result = minimize(_high_beta_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
