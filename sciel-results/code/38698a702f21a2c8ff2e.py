"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + spectral amplitude polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

# Pre-compute FFT sizes
L_CONV = 2 * N - 1
L_FFT = 1
while L_FFT < L_CONV:
    L_FFT *= 2
L_CORR = 1
while L_CORR < 3 * N:
    L_CORR *= 2

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    X = np.fft.rfft(x, n=L_FFT)
    C = np.fft.irfft(X * X, n=L_FFT)[:L_CONV]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Optimized with pre-computed FFT sizes.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation using pre-computed FFT size
    g_padded = np.zeros(L_CORR)
    g_padded[:L_CONV] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_CORR)
    G_f = np.fft.rfft(g_padded, n=L_CORR)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_CORR)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
    if mode == 0:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = np.ones(L_FFT // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_FFT / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_FFT * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_FFT // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = np.exp(-20 * (freqs * L_FFT / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_FFT)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_FFT // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_FFT // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_FFT)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """Optimize the phase of the spectrum while keeping magnitude fixed."""
    n_freq = L_FFT // 2 + 1
    
    X = np.fft.rfft(x0, n=L_FFT)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        
        G = np.fft.rfft(grad, n=L_FFT)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_FFT)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=40):
    """Optimize the amplitude spectrum while keeping phase fixed."""
    n_freq = L_FFT // 2 + 1
    
    X = np.fft.rfft(x0, n=L_FFT)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_FFT)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_FFT)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _peak_targeted_polish(x0, rng, max_iters=1200, step_size=0.01):
    """Identify dominant autoconvolution lags and apply targeted perturbations."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 15 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 10
        sorted_idx = np.argsort(-abs_ac)[:top_k]
        
        improved = False
        for k in sorted_idx:
            if _check_deadline():
                break
            
            # Vectorized contribution computation
            contrib = np.zeros(N)
            lo = max(0, k - N + 1)
            hi = min(k, N - 1)
            if lo <= hi:
                n_indices = np.arange(lo, hi + 1)
                kn_indices = k - n_indices
                contrib[n_indices] = 2.0 * best_x[kn_indices]
            
            norm_contrib = np.linalg.norm(contrib)
            if norm_contrib > 1e-10:
                direction = -contrib / norm_contrib
                
                for scale in [step_size, step_size * 0.5, step_size * 2.0]:
                    x_trial = best_x + scale * direction
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial.copy()
                        improved = True
                        break
        
        if improved:
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = min(0.05, step_size * 1.05)
    
    return best_x, best_val

def _high_precision_polish(x0, rng, max_iters=500):
    """Final high-precision polish with very high beta and tight tolerances."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for beta in [2000.0, 5000.0, 10000.0]:
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-12, 
                                                 'ftol': 1e-15, 'maxls': 20})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    # Final coordinate descent with very small steps
    step_size = 0.005
    no_improve = 0
    max_no_improve = 2000
    
    while no_improve < max_no_improve and not _check_deadline():
        r = rng.random()
        if r < 0.3:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.6:
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            freq = rng.integers(1, 4)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.01, step_size * 1.02)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
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
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
    ]
    
    # Add more diverse spectral initializations
    for mode in range(5):
        for _ in range(4):  # Increased from 3 to 4
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(10):  # Increased from 8 to 10
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Phase 1: L-BFGS with moderate beta (smooth landscape)
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        try:
            def obj_and_grad_func(x, _beta=50.0):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 
                                                 'ftol': 1e-12, 'maxls': 40})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=200, lr=0.015)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=30)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=1000, step_size=0.01)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-precision polish
    x_hp, val_hp = _high_precision_polish(best_x, rng, max_iters=500)
    if val_hp < best_val:
        best_val = val_hp
        best_x = x_hp.copy()
    
    return best_x
# EVOLVE-BLOCK-END
