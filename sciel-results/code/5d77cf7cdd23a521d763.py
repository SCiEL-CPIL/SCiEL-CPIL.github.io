"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _psd_objective(params, beta=50.0):
    """
    PSD-based objective: minimize peak of autoconv via power spectral density.
    params: [log_psd_0, ..., log_psd_{N-1}] where psd_k = |X_k|^2
    The autoconvolution is the inverse FFT of the PSD (shifted).
    We optimize in log-PSD space for better conditioning.
    """
    log_psd = params
    psd = np.exp(log_psd)
    
    # Reconstruct signal from PSD (take square root, assume zero phase)
    # PSD is |X|^2, so X = sqrt(psd) * exp(i*phase)
    # For simplicity, use zero phase: X = sqrt(psd)
    # But we need to handle the DC component and symmetry
    
    # The PSD from rfft has N/2+1 bins for real signals of length N
    # Actually for our problem, we work with the full spectrum
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Build full PSD from the optimization variables
    # We optimize over the first N bins (rfft of length N gives N//2+1 bins)
    # For simplicity, let's work directly with the autoconvolution
    
    # Alternative: parameterize x directly but use PSD insights for initialization
    # Let's stick with optimizing x but use a better surrogate
    
    s = np.sum(np.exp(log_psd / 2.0))  # This doesn't make sense for signed functions
    
    # Better approach: optimize x directly with a PSD-aware gradient
    return 1e10, np.zeros_like(params)

def _psd_objective_v2(x, beta=50.0):
    """
    Objective using PSD-aware smoothing.
    The autoconvolution c[n] = sum_k x[k]*x[k+n]
    In frequency: C(f) = |X(f)|^2
    So max|c| relates to the peak of the inverse FFT of |X|^2.
    
    We use a soft-max surrogate on |c| and compute gradient via chain rule through FFT.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx[j] = sum_n w[n] * sign(c[n]) * (x[j+n] + x[j-n])
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_init(rng, N, mode='cosine'):
    """
    Generate initializations based on spectral considerations.
    The goal is to make |X(f)|^2 as flat as possible (white noise spectrum)
    to minimize the peak of the autoconvolution.
    """
    t = np.linspace(0, 1, N)
    
    if mode == 'constant':
        return np.ones(N)
    elif mode == 'linear_down':
        return np.linspace(2.0, 0.5, N)
    elif mode == 'linear_up':
        return np.linspace(0.5, 2.0, N)
    elif mode == 'cosine':
        return 1.0 + 0.5 * np.cos(np.pi * t)
    elif mode == 'cosine_full':
        return 1.0 + np.cos(np.pi * t)
    elif mode == 'sin':
        return 1.0 + 0.5 * np.sin(np.pi * t)
    elif mode == 'exp_decay':
        return np.exp(-3.0 * t)
    elif mode == 'exp_growth':
        return np.exp(3.0 * t - 3.0)
    elif mode == 'random_smooth':
        freqs = rng.uniform(0.1, 4.0, size=6)
        phases = rng.uniform(0, 2*np.pi, size=6)
        amps = rng.uniform(0.1, 0.6, size=6)
        return 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
    elif mode == 'geometric':
        r = rng.uniform(0.5, 1.5)
        return r ** np.arange(N)
    else:
        return np.ones(N)

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

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Diverse spectral-aware initializations
    rng_init = np.random.default_rng(123)
    modes = ['constant', 'linear_down', 'linear_up', 'cosine', 'cosine_full',
             'sin', 'exp_decay', 'exp_growth']
    
    for mode in modes:
        x0 = _spectral_init(rng_init, N, mode)
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Random smooth initializations
    for _ in range(30):
        if _check_deadline():
            break
        x0 = _spectral_init(rng_init, N, 'random_smooth')
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Geometric sequences
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        if _check_deadline():
            break
        x0 = r ** np.arange(N)
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate (multiple betas)
    for beta in [30.0, 50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            return _psd_objective_v2(x, beta=b)
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 250, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Multi-start refinement
    rng = np.random.default_rng(42)
    for restart in range(10):
        if _check_deadline():
            break
        
        # Perturb the current best
        x0 = best_x + rng.normal(0, 0.1, size=N)
        
        for beta in [50.0, 100.0]:
            if _check_deadline():
                break
            
            def _obj_restart(x, b=beta):
                return _psd_objective_v2(x, beta=b)
            
            try:
                result = minimize(_obj_restart, x0, method='L-BFGS-B', jac=True,
                                options={'maxiter': 150, 'gtol': 1e-6, 'ftol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Final high-beta L-BFGS polish
    def _final_obj(x):
        return _psd_objective_v2(x, beta=300.0)
    
    try:
        result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Random perturbation refinement with adaptive step size
    rng = np.random.default_rng(99)
    step_size = 0.03
    stall_count = 0
    max_iters = 50000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9995)
        else:
            stall_count += 1
            if stall_count % 3000 == 0:
                step_size = min(0.3, step_size * 1.08)
    
    # Final coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=2000, step_scale=0.02)
    
    # Very final polish
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
