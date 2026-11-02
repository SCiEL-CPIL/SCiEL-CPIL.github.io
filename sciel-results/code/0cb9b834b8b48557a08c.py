"""PSD shaping: minimize L-inf norm of autoconvolution via spectral amplitude optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
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
    Uses the efficient cross-correlation form: grad_x = 2 * corr(grad_ac, x)
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        return m, np.zeros(N)
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] for valid k-n
    # This is 2 * cross_correlation(grad_ac_padded, x)
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
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

def _psd_shaping(x0, rng, max_iters=500, lr_init=0.005, beta=50.0):
    """
    Spectral amplitude optimization: minimize max|ac| by shaping the PSD.
    In frequency domain, ac is the inverse FFT of |X|^2 (power spectral density).
    We optimize the amplitude spectrum to flatten the PSD, reducing peak autoconvolution.
    Uses projected gradient descent in the frequency domain.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Initialize amplitude spectrum
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0.0
    mag[0] = max(mag[0], 1e-3)
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    lr = lr_init
    no_improve = 0
    max_no_improve = 100
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Current time-domain signal
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x = np.fft.irfft(spectrum, n=L_fft)[:N]
        s = np.sum(x)
        if s < 0:
            x = -x
            phase = -phase
            phase[0] = 0.0
        
        # Compute gradient in time domain
        val, grad = _objective_and_grad(x, beta)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Transform gradient to frequency domain
        G = np.fft.rfft(grad, n=L_fft)
        
        # Gradient of objective w.r.t. amplitude:
        # d(obj)/d(mag[k]) = Re[conj(G[k]) * exp(1j*phase[k])] for k != 0, Nyquist
        # For k=0: d(obj)/d(mag[0]) = Re[G[0]]
        # For Nyquist (if L_fft even): d(obj)/d(mag[n_freq-1]) = Re[G[n_freq-1]]
        
        amp_grad = np.zeros(n_freq)
        for k in range(1, n_freq):
            if k == L_fft // 2:
                amp_grad[k] = np.real(G[k])
            else:
                amp_grad[k] = np.real(np.conj(G[k]) * np.exp(1j * phase[k]))
        amp_grad[0] = np.real(G[0])
        
        # Update amplitude with projection (non-negative)
        mag -= lr * amp_grad
        mag = np.maximum(mag, 0.0)
        
        # Adaptive learning rate
        if no_improve == 0 and it % 10 == 0 and it > 0:
            lr = min(0.05, lr * 1.05)
        elif no_improve > 20:
            lr = max(0.0005, lr * 0.9)
    
    return best_x, best_val

def _phase_polish(x0, rng, max_iters=300, lr_init=0.01):
    """
    Phase-only optimization: keep amplitude fixed, optimize phase.
    This can find better local minima in the phase space.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0.0
    mag[0] = max(mag[0], 1e-3)
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    lr = lr_init
    no_improve = 0
    max_no_improve = 80
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x = np.fft.irfft(spectrum, n=L_fft)[:N]
        s = np.sum(x)
        if s < 0:
            x = -x
            phase = -phase
            phase[0] = 0.0
        
        val, grad = _objective_and_grad(x, 50.0)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        G = np.fft.rfft(grad, n=L_fft)
        
        # Phase gradient: d(obj)/d(phase[k]) = Im[conj(G[k]) * spectrum[k]]
        phase_grad = np.zeros(n_freq)
        for k in range(1, n_freq):
            if k == L_fft // 2:
                continue
            phase_grad[k] = np.imag(np.conj(G[k]) * spectrum[k])
        phase_grad[0] = 0.0
        
        phase -= lr * phase_grad
        
        if no_improve == 0 and it % 10 == 0 and it > 0:
            lr = min(0.05, lr * 1.05)
        elif no_improve > 15:
            lr = max(0.001, lr * 0.85)
    
    return best_x, best_val

def _time_domain_lbfgs(x0, beta, maxiter=200):
    """
    L-BFGS optimization in time domain using soft-max surrogate.
    """
    def obj_and_grad_func(x):
        return _objective_and_grad(x, beta)
    
    try:
        result = minimize(obj_and_grad_func, x0, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-12, 'ftol': 1e-14})
        return result.x, result.fun
    except Exception:
        return x0, _objective_only(x0)

def _coordinate_descent(x0, rng, max_iters=5000):
    """
    Coordinate descent refinement with random perturbations.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    step_size = 0.01
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        n_perturb = rng.integers(2, 15)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size = min(0.05, step_size * 1.1)
        
        if no_improve > max_no_improve:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Phase 1: PSD shaping from multiple random initializations
    for restart in range(5):
        if _check_deadline():
            break
        
        # Random initial signal
        x_init = rng.normal(0, 1, size=N)
        # Ensure non-negative sum
        if np.sum(x_init) < 0:
            x_init = -x_init
        
        x_psd, val_psd = _psd_shaping(x_init, rng, max_iters=300)
        
        if val_psd < best_val:
            best_val = val_psd
            best_x = x_psd.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Phase polish on the best PSD-shaped solution
    x_phase, val_phase = _phase_polish(best_x, rng, max_iters=200)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: L-BFGS refinement in time domain with multiple beta values
    for beta in [30.0, 50.0, 100.0]:
        if _check_deadline():
            break
        x_lbfgs, val_lbfgs = _time_domain_lbfgs(best_x, beta, maxiter=150)
        if val_lbfgs < best_val:
            best_val = val_lbfgs
            best_x = x_lbfgs.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent for final polish
    x_cd, val_cd = _coordinate_descent(best_x, rng, max_iters=3000)
    if val_cd < best_val:
        best_val = val_cd
        best_x = x_cd.copy()
    
    return best_x
# EVOLVE-BLOCK-END
