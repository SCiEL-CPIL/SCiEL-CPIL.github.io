"""PSD-domain optimization: minimize max(|autoconv|)/(sum x)^2 by optimizing spectral shape."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
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

def _psd_to_signal(psd):
    """
    Reconstruct real time-domain signal from PSD with zero phase.
    psd is the real, non-negative, symmetric power spectral density.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # X[k] = sqrt(PSD[k]) with zero phase
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2+1] = np.sqrt(np.maximum(psd, 0))
    for k in range(1, L_fft//2+1):
        X[L_fft-k] = X[k]
    
    x = np.fft.ifft(X).real[:N]
    return x

def _psd_objective_and_grad(psd, beta=100.0):
    """
    Smooth objective and gradient for PSD optimization.
    Minimizes smooth_max(|ifft(psd)|) / psd[0]
    where psd[0] = (sum x)^2.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Compute autoconvolution from PSD
    ac = np.fft.irfft(psd, n=L_fft)[:L]
    abs_ac = np.abs(ac)
    
    dc = psd[0]
    if dc < 1e-15:
        return 1e10, np.zeros_like(psd)
    
    # Smooth max via log-sum-exp
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / dc
    
    # Gradient computation
    c = weights * np.sign(ac)
    
    # d(smooth_max)/dpsd[k] = (1/L_fft) * sum_j c[j] * exp(2*pi*i*j*k/L_fft)
    # For real c: = (1/L_fft) * conj(fft(c))[k]
    fft_c = np.fft.fft(c, n=L_fft)
    grad_smooth_max_psd = np.real(fft_c.conj()) / L_fft
    
    # Gradient of val = smooth_max / dc
    grad = grad_smooth_max_psd / dc
    grad[0] = (grad_smooth_max_psd[0] * dc - smooth_max) / (dc * dc)
    
    # Enforce Hermitian symmetry in gradient for real PSD
    half = L_fft // 2
    for k in range(1, half + 1):
        idx = L_fft - k
        if idx < len(grad):
            grad[k] = (grad[k] + grad[idx]) / 2
            if idx < len(grad):
                grad[idx] = grad[k]
    
    return val, grad[:L_fft//2+1]

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: PSD-domain optimization with multiple initializations
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_psd = L_fft // 2 + 1
    
    # Create various PSD initializations
    t = np.linspace(0, 1, n_psd)
    psd_inits = []
    
    # Flat PSD (white noise)
    psd_inits.append(np.ones(n_psd))
    
    # Low-pass filters with various cutoffs
    for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        psd_inits.append(1.0 / (1.0 + (t / cutoff) ** 2))
    
    # High-pass filters
    for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5]:
        psd_inits.append((t / cutoff) ** 2 / (1.0 + (t / cutoff) ** 2))
    
    # Band-pass filters
    for center, width in [(0.3, 0.1), (0.5, 0.15), (0.7, 0.2), (0.4, 0.2), (0.6, 0.25)]:
        psd_inits.append(np.exp(-((t - center) / width) ** 2))
    
    # Power-law PSDs
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        psd_inits.append((t + 0.01) ** (-alpha))
    
    # Cosine-shaped PSDs
    for freq in [1, 2, 3, 4]:
        psd_inits.append(0.5 + 0.5 * np.cos(2 * np.pi * freq * t))
    
    # Sine-shaped PSDs
    for freq in [1, 2, 3]:
        psd_inits.append(0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
    
    # Multi-peak PSDs
    for peaks in [(0.2, 0.6), (0.3, 0.7), (0.4, 0.8), (0.25, 0.5, 0.75)]:
        psd_peak = np.zeros(n_psd)
        for p in peaks:
            psd_peak += np.exp(-((t - p) / 0.08) ** 2)
        psd_inits.append(psd_peak)
    
    # Evaluate each initialization
    for psd0 in psd_inits:
        if _check_deadline():
            break
        
        # Ensure PSD is non-negative and properly scaled
        psd0 = np.maximum(psd0, 0)
        if psd0[0] < 1e-10:
            psd0[0] = 1.0
        
        # Normalize so that DC component gives a reasonable signal
        psd0 = psd0 / psd0[0] * 100.0
        
        x0 = _psd_to_signal(psd0)
        val0 = _objective(x0)
        
        if val0 < best_val:
            best_val = val0
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: PSD optimization from best time-domain solution
    # Convert best_x to PSD
    X_best = np.fft.rfft(best_x, n=L_fft)
    psd_best = np.abs(X_best) ** 2
    
    # Optimize PSD with multiple betas
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if _check_deadline():
            break
        
        def _psd_obj_grad(p, b=beta):
            val, grad = _psd_objective_and_grad(p, beta=b)
            return val, grad
        
        try:
            result = minimize(_psd_obj_grad, psd_best, method='L-BFGS-B',
                             jac=True, options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-10})
            
            # Ensure PSD is non-negative
            psd_result = np.maximum(result.x, 0)
            x_result = _psd_to_signal(psd_result)
            val_result = _objective(x_result)
            
            if val_result < best_val:
                best_val = val_result
                best_x = x_result.copy()
                psd_best = psd_result.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Multi-start PSD optimization from perturbed PSDs
    rng = np.random.default_rng(42)
    for restart in range(10):
        if _check_deadline():
            break
        
        # Perturb the current best PSD
        for scale in [0.1, 0.3, 0.5, 1.0]:
            if _check_deadline():
                break
            
            psd_pert = psd_best + rng.normal(0, scale * psd_best[0], size=n_psd)
            psd_pert = np.maximum(psd_pert, 0)
            if psd_pert[0] < 1e-10:
                psd_pert[0] = 1.0
            
            x_pert = _psd_to_signal(psd_pert)
            val_pert = _objective(x_pert)
            
            if val_pert < best_val:
                best_val = val_pert
                best_x = x_pert.copy()
                psd_best = psd_pert.copy()
            
            # Refine from this PSD
            for beta in [100.0, 200.0]:
                if _check_deadline():
                    break
                
                def _psd_obj_grad2(p, b=beta):
                    val, grad = _psd_objective_and_grad(p, beta=b)
                    return val, grad
                
                try:
                    result = minimize(_psd_obj_grad2, psd_best, method='L-BFGS-B',
                                     jac=True, options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10})
                    
                    psd_result = np.maximum(result.x, 0)
                    x_result = _psd_to_signal(psd_result)
                    val_result = _objective(x_result)
                    
                    if val_result < best_val:
                        best_val = val_result
                        best_x = x_result.copy()
                        psd_best = psd_result.copy()
                except Exception:
                    pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Time-domain refinement from best PSD solution
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def _time_obj_grad(x, b=beta):
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e10, np.zeros(N)
            ac = _autoconv_fft(x)
            abs_ac = np.abs(ac)
            
            max_val = np.max(abs_ac)
            if max_val < 1e-15:
                smooth_max = max_val
                weights = np.ones_like(abs_ac) / len(abs_ac)
            else:
                shifted = abs_ac - max_val
                exp_shifted = np.exp(b * shifted)
                sum_exp = np.sum(exp_shifted)
                smooth_max = (max_val + np.log(sum_exp)) / b
                weights = exp_shifted / sum_exp
            
            val = smooth_max / (s * s)
            
            c = weights * np.sign(ac)
            L_conv = 2 * N - 1
            L_fft2 = 1
            while L_fft2 < 2 * L_conv:
                L_fft2 *= 2
            
            x_pad = np.zeros(L_fft2)
            x_pad[:N] = x
            c_pad = np.zeros(L_fft2)
            c_pad[:L_conv] = c
            
            X_fft = np.fft.rfft(x_pad)
            C_fft = np.fft.rfft(c_pad)
            
            cross1 = np.fft.irfft(C_fft * np.conj(X_fft), n=L_fft2)[:N]
            cross2 = np.fft.irfft(C_fft * X_fft, n=L_fft2)[:N]
            
            grad_smooth_max = cross1 + cross2
            grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
            
            return val, grad
        
        try:
            result = minimize(_time_obj_grad, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-10})
            
            val_result = _objective(result.x)
            if val_result < best_val:
                best_val = val_result
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
