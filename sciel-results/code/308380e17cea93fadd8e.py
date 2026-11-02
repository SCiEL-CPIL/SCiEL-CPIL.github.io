import numpy as np
import time

N = 400
TIME_LIMIT = 145  # safety margin before 150s kill

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
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_obj_grad(x, tau=0.1):
    """
    Compute soft-max approximation of max(|ac|)/(sum x)^2 and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft-max weights
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    soft_max = m + tau * np.log(w_sum)
    val = soft_max / (s * s)
    
    # Gradient of soft_max w.r.t. x
    # v[i] = weights[i] * sign(ac[i])
    v = weights * np.sign(ac)
    
    # Use FFT for gradient computation
    L_fft = 1
    while L_fft < L + N:
        L_fft *= 2
    
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    V = np.fft.rfft(v_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # Term 1: sum_i v[i] * x[i-k] (cross-correlation)
    corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
    
    # Term 2: sum_i v[i] * x[k-i] (convolution)
    conv2 = np.fft.irfft(V * X, n=L_fft)
    
    grad_sm = (corr1[:N] + conv2[:N])
    
    # Gradient of objective = soft_max / s^2
    grad_obj = grad_sm / (s * s) - 2.0 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    START_TIME = time.time()
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
    
    for _ in range(30):
        initializations.append(rng.normal(1.0, 0.5, size=N))
    
    # Also try some structured patterns
    for period in [4, 8, 16, 32]:
        pattern = np.ones(N)
        pattern[::2] = -1.0
        initializations.append(pattern)
    
    for x0 in initializations:
        if not _time_ok():
            break
        if np.sum(x0) < 0:
            x0 = -x0
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Phase 2: Gradient descent with smooth surrogate
    tau = 0.05
    lr = 0.1
    for gd_iter in range(50000):
        if not _time_ok():
            break
        
        val_s, grad = _smooth_obj_grad(best_x, tau)
        if not np.isfinite(val_s):
            break
        
        actual_val = _objective(best_x)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-10:
            break
        
        step = grad / (norm_grad + 1e-10)
        
        alpha = lr
        improved = False
        for _ in range(15):
            x_trial = best_x - alpha * step
            trial_val = _objective(x_trial)
            if trial_val < actual_val * (1 - 1e-12):
                best_x = x_trial.copy()
                best_val = trial_val
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.7
            if lr < 1e-10:
                break
        else:
            lr = min(0.5, lr * 1.15)
        
        if gd_iter % 200 == 0 and tau > 0.005:
            tau *= 0.9
    
    # Phase 3: Peak-targeted refinement
    # Focus on the dominant lags in autoconvolution
    if _time_ok():
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        # Get top few lags
        top_k = 5
        top_indices = np.argsort(abs_ac)[-top_k:]
        
        # Do targeted gradient steps on these peaks
        tau_peak = 0.01
        for peak_iter in range(20000):
            if not _time_ok():
                break
            
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            s = np.sum(best_x)
            if abs(s) < 1e-10:
                break
            
            # Soft-max over just the top peaks
            m = np.max(abs_ac)
            shifted = (abs_ac - m) / tau_peak
            shifted = np.clip(shifted, -50, 0)
            weights = np.exp(shifted)
            w_sum = np.sum(weights)
            if w_sum < 1e-15:
                break
            weights /= w_sum
            
            soft_max = m + tau_peak * np.log(w_sum)
            v = weights * np.sign(ac)
            
            L = len(ac)
            L_fft = 1
            while L_fft < L + N:
                L_fft *= 2
            
            v_pad = np.zeros(L_fft)
            v_pad[:L] = v
            x_pad = np.zeros(L_fft)
            x_pad[:N] = best_x
            
            V = np.fft.rfft(v_pad, n=L_fft)
            X = np.fft.rfft(x_pad, n=L_fft)
            
            corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
            conv2 = np.fft.irfft(V * X, n=L_fft)
            grad_sm = (corr1[:N] + conv2[:N])
            
            grad_obj = grad_sm / (s * s) - 2.0 * soft_max / (s * s * s)
            
            norm_grad = np.linalg.norm(grad_obj)
            if norm_grad < 1e-12:
                break
            
            step = grad_obj / (norm_grad + 1e-10)
            lr_peak = 0.05
            improved = False
            for _ in range(10):
                x_trial = best_x - lr_peak * step
                trial_val = _objective(x_trial)
                if trial_val < best_val * (1 - 1e-12):
                    best_x = x_trial.copy()
                    best_val = trial_val
                    improved = True
                    break
                lr_peak *= 0.5
            
            if not improved:
                break
    
    # Phase 4: Fine-tuning with small perturbations
    step_size = 0.01
    stall_count = 0
    for iteration in range(300000):
        if not _time_ok():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.0005, step_size * 0.9995)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.3, step_size * 1.03)
    
    return best_x
