"""Model-and-optimize: spectral phase retrieval + L-BFGS + adaptive polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs
    
    # Gradient w.r.t. x via FFT-based correlation
    g = grad_ac  # length 2N-1
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g, n=L_corr)
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_corr)
    term1 = corr[:N]
    
    grad_x = term1 + term2
    return soft_val, grad_x

def _objective_only(x, beta):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init_phase_optimized(rng, mode=0, iters=8):
    """
    Generate initial x via spectral phase retrieval, then refine phase 
    to minimize max|ac| (PAPR-like metric).
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # 1. Generate initial magnitude spectrum
    if mode == 0:
        # Flat-top with taper
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Raised cosine
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    else:
        # Random structure
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    
    # 2. Random phase
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0  # DC real
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x = np.fft.irfft(spectrum, n=L_fft)[:N]
    
    # Normalize sum to 1 for phase optimization stability (scale back later)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return x
    x_norm = x / s
    
    # 3. Phase optimization: minimize max|ac|
    for _ in range(iters):
        if _check_deadline():
            break
        # Compute gradient of soft-max ac
        beta = 50.0
        val, grad = _soft_max_and_grad(x_norm, beta)
        
        # Update x_norm
        step = 1.0 / (beta * np.max(np.abs(x_norm)) + 1e-6)
        x_new = x_norm - step * grad
        
        # Re-project to fixed magnitude spectrum
        X_new = np.fft.rfft(x_new, n=L_fft)
        X_new = mag * (X_new / np.maximum(np.abs(X_new), 1e-10))
        X_new[0] = mag[0] # Keep DC real
        
        x_norm = np.fft.irfft(X_new, n=L_fft)[:N]
        
        # Re-normalize sum to 1
        s_new = np.sum(x_norm)
        if abs(s_new) < 1e-10:
            break
        x_norm = x_norm / s_new

    return x_norm

def _amplitude_optimize(x_init, iters=10):
    """
    Optimize the amplitude spectrum (keeping phase fixed) to reduce max|ac|.
    This is a gradient-based approach in the frequency domain.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x_init, n=L_fft)
    mag = np.abs(X)
    phase = np.angle(X)
    mag[0] = np.abs(X[0])
    
    for i in range(iters):
        if _check_deadline():
            break
        
        # Reconstruct x from current mag and phase
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]  # Keep DC real
        x = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        x_norm = x / s
        
        # Compute gradient of soft-max ac w.r.t. x
        beta = 100.0
        val, grad_x = _soft_max_and_grad(x_norm, beta)
        
        # Compute gradient w.r.t. amplitude in frequency domain
        # d x / d mag_k = (1/L_fft) * exp(i*phase_k) * exp(-i*2*pi*k*n/L_fft)
        # We use FFT-based approach: grad_mag = Re[conj(X) * FFT(grad_x)] / L_fft
        # Actually, the relationship is: x = IFFT(mag * exp(i*phase))
        # dx/dmag_k = (1/L_fft) * exp(i*phase_k) * w_k where w_k are DFT basis
        # In practice: grad_mag = Re[conj(X) * FFT_padded(grad_x)] / L_fft
        
        # More precisely: if x = IFFT(M), M = mag * exp(i*phase)
        # Then d<grad_x, x>/dmag_k = Re[conj(exp(i*phase_k)) * conj(FFT_basis_k)]
        # Simpler: compute FFT of grad_x, take real part after phase alignment
        G = np.fft.rfft(grad_x, n=L_fft)
        # Gradient w.r.t. amplitude: Re[conj(phase_factor) * G] where phase_factor = exp(i*phase)
        # Actually: d<grad, IFFT(M)>/d|X_k| = Re[conj(X_k/|X_k|) * conj(FFT(grad)_k)] / L_fft
        # Let's just use: grad_mag = Re[conj(X) * G] / L_fft
        # This works because X = mag * exp(i*phase), so conj(X)*G / mag = conj(exp(i*phase))*G
        
        # Simpler approach: finite difference or direct gradient
        # grad_mag[k] = (2/L_fft) * Re[conj(X[k]) * G[k]] for k>0, (1/L_fft)*Re[G[0]] for k=0
        grad_mag = np.zeros_like(mag)
        if L_fft > 1:
            grad_mag[1:] = (2.0 / L_fft) * np.real(np.conj(X[1:]) * G[1:])
        grad_mag[0] = np.real(G[0]) / L_fft
        
        # Also need to account for the 1/s^2 factor and the -2*soft_val/s^3 term
        # But for amplitude optimization, we just minimize the soft-max ac directly
        # (the normalization is handled by re-normalizing after each step)
        
        # Gradient descent on amplitude
        step_size = 1.0 / (beta * np.max(mag) + 1e-6)
        mag = mag - step_size * grad_mag
        mag = np.maximum(mag, 0.01)  # Keep amplitudes positive
        mag[0] = np.maximum(mag[0], 0.01)
        
        # Re-normalize to keep sum(x) = 1
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_new = np.fft.irfft(spectrum, n=L_fft)[:N]
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            x_norm = x_new / s_new
        else:
            x_norm = x_new
    
    return x_norm

def construct():
    best_x = np.ones(N)
    best_val = _objective_and_grad(best_x, 50.0)[0]
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
    ]
    
    # Add spectral initializations with phase optimization
    for mode in range(3):
        for _ in range(4):
            if _check_deadline():
                break
            initial_points.append(_spectral_init_phase_optimized(rng, mode, iters=8))
    
    # Add random initializations
    for _ in range(6):
        if _check_deadline():
            break
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # L-BFGS optimization from each initial point
    # Strategy: Start with moderate beta, then refine
    betas = [50.0, 100.0]
    
    candidates = []
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 120, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                
                # Keep track of top candidates for final polish
                candidates.append((result.fun, result.x))
                
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Sort candidates and polish the top few
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:6]]
    
    # Amplitude optimization on top candidates
    for x_init in top_candidates:
        if _check_deadline():
            break
        try:
            x_amp = _amplitude_optimize(x_init, iters=8)
            val_amp = _objective_only(x_amp, 100.0)
            if val_amp < best_val:
                best_val = val_amp
                best_x = x_amp.copy()
        except Exception:
            pass
    
    # Final Polish with high beta
    for x_init in top_candidates:
        if _check_deadline():
            break
        
        for beta in [200.0, 500.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_init, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 250, 'gtol': 1e-10, 'ftol': 1e-12})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    # Final deterministic local search (Coordinate Descent style)
    # This is faster than random perturbation and more directed.
    if not _check_deadline():
        step_size = 0.01
        beta_fast = 500.0
        no_improve = 0
        max_no_improve = 3000
        
        # Use a simple cyclic coordinate descent
        while no_improve < max_no_improve and not _check_deadline():
            improved = False
            # Try perturbing a small random subset of indices
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
            val = _objective_only(x_trial, beta_fast)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                step_size = max(0.001, step_size * 0.95)
                no_improve = 0
                improved = True
            else:
                no_improve += 1
                step_size = min(0.05, step_size * 1.05)
                
            if not improved and no_improve > 500:
                # If stuck, try a larger jump
                perturbation = np.zeros(N)
                indices = rng.choice(N, size=10, replace=False)
                perturbation[indices] = rng.normal(0, 0.05, size=10)
                x_trial = best_x + perturbation
                val = _objective_only(x_trial, beta_fast)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    no_improve = 0
                    step_size = 0.01
                else:
                    no_improve += 1

    return best_x
# EVOLVE-BLOCK-END
