"""Spectral amplitude optimization: minimize L-inf PSD with unit-DC constraint, then time-domain polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
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

def _psd_to_signal(psd):
    """
    Convert a one-sided PSD (length N) to a real time-domain signal.
    Uses zero-phase reconstruction with symmetric PSD.
    """
    psd = np.maximum(psd, 0)
    
    # Build full symmetric PSD for real signal
    if N % 2 == 0:
        psd_full = np.zeros(N)
        psd_full[0] = psd[0]
        psd_full[N//2] = psd[N//2]
        for k in range(1, N//2):
            psd_full[k] = psd[k]
            psd_full[N-k] = psd[k]
    else:
        psd_full = np.zeros(N)
        psd_full[0] = psd[0]
        for k in range(1, (N+1)//2):
            psd_full[k] = psd[k]
            psd_full[N-k] = psd[k]
    
    X = np.sqrt(psd_full).astype(complex)
    x = np.fft.ifft(X).real
    return x

def _spectral_projected_gradient_descent(psd_init, n_iters=2000, lr=0.01):
    """
    Projected gradient descent in PSD space to minimize L-inf(psd) subject to sum(x)=1.
    The gradient of L-inf(psd) w.r.t. psd is 1 at the max bins, 0 elsewhere.
    We use a smooth approximation: softmax over psd values.
    """
    psd = psd_init.copy()
    n_bins = len(psd)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Compute current signal and sum
        x = _psd_to_signal(psd)
        s = np.sum(x)
        
        if abs(s) < 1e-10:
            # Perturb to avoid zero sum
            psd[0] += 0.1
            continue
        
        # Normalize so sum(x) = 1
        psd = psd / (s * s)  # scaling PSD by c scales x by c, sum by c
        # Actually: if x -> c*x, then PSD -> c^2*PSD, sum -> c*sum
        # So to get sum=1 from current sum s, we need c=1/s, so PSD_new = PSD/s^2
        
        x = _psd_to_signal(psd)
        s = np.sum(x)
        
        # Smooth L-inf surrogate: use log-sum-exp
        beta = 50.0
        shifted = beta * (psd - np.max(psd))
        exp_shifted = np.exp(shifted)
        lse = np.max(psd) + (1.0/beta) * np.log(np.sum(exp_shifted))
        
        # Gradient of LSE w.r.t. psd
        weights = exp_shifted / np.sum(exp_shifted)
        grad = weights
        
        # Gradient descent step
        psd = psd - lr * grad
        
        # Project: ensure non-negative
        psd = np.maximum(psd, 0)
        
        # Re-normalize for unit DC
        x = _psd_to_signal(psd)
        s = np.sum(x)
        if abs(s) > 1e-10:
            psd = psd / (s * s)
    
    return psd

def _vectorized_peak_targeted_polish(x, n_iters=50000, beta=50.0):
    """
    Vectorized peak-targeted polish: track top-K peaks, apply structured perturbations.
    Much faster than per-lag Python loops.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    rng = np.random.default_rng(123)
    step_size = 0.1
    K = 10  # number of top peaks to track
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Compute autoconv
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        # Get top-K peak indices
        top_k_idx = np.argpartition(abs_ac, -K)[-K:]
        
        # Soft-max weights over top peaks
        beta_local = beta
        shifted = beta_local * (abs_ac[top_k_idx] - abs_ac[top_k_idx].max())
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        
        # Build gradient-like direction: push down at peaks, up elsewhere
        # For each peak index, the contribution to x is via convolution
        # Simplified: use random structured perturbations weighted by peak positions
        
        # Generate structured perturbation
        perturbation = rng.normal(0, step_size, size=N)
        
        # Add bias: reduce energy at frequencies corresponding to peak lags
        # Peak at lag m corresponds to frequency modulation
        for idx in top_k_idx:
            # Lag idx corresponds to a shift; add a cosine perturbation
            freq = idx / N
            cos_pert = np.cos(2 * np.pi * freq * np.arange(N))
            perturbation -= 0.1 * weights[list(top_k_idx).index(idx)] * cos_pert
        
        x_trial = best_x + perturbation
        val = _objective(x_trial)
        
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.999)
        else:
            step_size = min(0.5, step_size * 1.005)
    
    return best_x, best_val

def construct():
    # Phase 1: Spectral design - start with flat PSD
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Flat PSD: all bins equal
    psd_flat = np.ones(N)
    x_flat = _psd_to_signal(psd_flat)
    s_flat = np.sum(x_flat)
    if abs(s_flat) > 1e-10:
        x_flat = x_flat / s_flat  # normalize to unit sum
        val_flat = _objective(x_flat)
        if val_flat < best_val:
            best_val = val_flat
            best_x = x_flat.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Projected gradient descent in PSD space
    # Start from flat PSD and various initializations
    psd_initials = [
        np.ones(N),  # flat
        np.exp(-np.arange(N) / (N * 0.5)),  # exponential decay
        np.exp(-np.arange(N)**2 / (N * N)),  # Gaussian
        np.cos(np.pi * np.arange(N) / N) ** 2,  # raised cosine
    ]
    
    for psd_init in psd_initials:
        if _check_deadline():
            break
        
        try:
            psd_opt = _spectral_projected_gradient_descent(psd_init, n_iters=1000, lr=0.005)
            x_opt = _psd_to_signal(psd_opt)
            s_opt = np.sum(x_opt)
            if abs(s_opt) > 1e-10:
                x_opt = x_opt / s_opt
                val_opt = _objective(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: L-BFGS-B with soft-max surrogate in time domain
    def _objective_grad_smooth(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(x)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        # Soft-max with high beta for sharper gradient
        beta = 80.0
        shifted = beta * (abs_ac - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
        val = soft_max / (s * s)
        
        # Gradient
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
        
        ds_dx = np.ones(N)
        grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * ds_dx
        
        return val, grad
    
    try:
        result = minimize(_objective_grad_smooth, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Vectorized peak-targeted polish
    try:
        best_x, best_val = _vectorized_peak_targeted_polish(best_x, n_iters=30000, beta=60.0)
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Final L-BFGS-B polish
    try:
        result = minimize(_objective_grad_smooth, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-16})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final exact evaluation
    final_val = _objective(best_x)
    if final_val < best_val:
        best_val = final_val
    
    return best_x
