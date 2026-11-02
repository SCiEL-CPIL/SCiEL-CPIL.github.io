"""Spectral-design + FFT-accelerated minimax for signed step functions minimizing max|autoconv|/(sum x)^2."""
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

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _grad_minimax_fft(x, q, ac):
    """
    Compute gradient of E_q[|ac|]/(sum x)^2 w.r.t. x using FFT.
    Returns (grad, expected_abs, s, s_sq).
    """
    L = 2 * N - 1
    s = np.sum(x)
    if abs(s) < 1e-10:
        s = 1e-10
    s_sq = s * s
    
    abs_ac = np.abs(ac)
    w = q * np.sign(ac)  # length L, w[k] = q[k]*sign(ac[k])
    
    L_fft = 1
    while L_fft < N + L:
        L_fft *= 2
    
    F = np.fft.rfft(w, n=L_fft)
    G = np.fft.rfft(x, n=L_fft)
    
    conv = np.fft.irfft(F * G, n=L_fft)[:N]
    corr = np.fft.irfft(np.conj(F) * G, n=L_fft)[:N]
    
    grad_x_raw = corr + conv
    
    expected_abs = np.sum(q * abs_ac)
    grad_x = grad_x_raw / s_sq - 2.0 * expected_abs * x / (s_sq * s)
    
    return grad_x, expected_abs, s, s_sq

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _minimax_optimize(x_init, max_time=100):
    """
    Run the minimax game from initial x_init with FFT-accelerated gradients.
    """
    L = 2 * N - 1
    q = np.ones(L) / L
    lr_x = 0.005
    
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    start = time.time()
    
    while time.time() - start < max_time:
        if _check_deadline():
            break
        
        for it in range(30):
            if _check_deadline():
                break
            
            ac = _autoconv_fft(x)
            abs_ac = np.abs(ac)
            
            tau = 5.0
            log_q = abs_ac / tau
            log_q -= np.max(log_q)
            q_new = np.exp(log_q)
            q_new /= np.sum(q_new)
            q = 0.7 * q + 0.3 * q_new
            
            grad_x, _, _, _ = _grad_minimax_fft(x, q, ac)
            x = x - lr_x * grad_x
        
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        lr_x *= 0.95
    
    return best_x, best_val

def _lbfgs_optimize(x_init, max_time=30):
    """
    Use L-BFGS-B with soft-max surrogate to optimize.
    """
    def fun(x):
        val, grad = _objective_and_grad(x, 50.0)
        return val, grad
    
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    start = time.time()
    
    try:
        res = minimize(fun, x, jac=True, method='L-BFGS-B',
                       options={'maxiter': 50, 'gtol': 1e-4})
        
        if res.x is not None:
            val = _objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_val

def _coordinate_descent_polish(x, max_iters=150):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective(best_x)
    
    deltas = [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            for d in deltas:
                trial = best_x.copy()
                trial[j] += d
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
    
    for mode in range(5):
        for _ in range(3):
            if _check_deadline():
                break
            x_spec = _spectral_init(rng, mode)
            initial_points.append(x_spec)
    
    # Allocate time budget among restarts
    time_per_restart = 12
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        try:
            # First try L-BFGS with soft-max
            x_lbfgs, val_lbfgs = _lbfgs_optimize(x0, max_time=15)
            if val_lbfgs < best_val:
                best_val = val_lbfgs
                best_x = x_lbfgs.copy()
            
            # Then try minimax
            if not _check_deadline():
                x_minimax, val_minimax = _minimax_optimize(x_lbfgs, max_time=time_per_restart)
                if val_minimax < best_val:
                    best_val = val_minimax
                    best_x = x_minimax.copy()
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=100)
    
    return best_x
# EVOLVE-BLOCK-END
