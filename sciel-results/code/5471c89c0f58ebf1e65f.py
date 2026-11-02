import numpy as np
import time
from scipy.optimize import minimize

N = 400

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

def _soft_max_and_grad(x, tau):
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
    v = weights * np.sign(ac)
    
    # Use FFT for efficiency
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

def _peak_targeted_polish(x, max_iters=5000, time_limit=10.0):
    """
    Targeted polish: identify dominant autoconvolution lags and apply
    targeted perturbations to reduce them.
    """
    start = time.time()
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        L = len(ac)
        
        # Find the top few dominant lags
        top_k = min(10, L)
        top_indices = np.argsort(abs_ac)[-top_k:]
        
        # For each dominant lag, try to reduce it by adjusting x
        improved = False
        for k in top_indices:
            if time.time() - start > time_limit:
                break
            
            # The autoconvolution at lag k is:
            # ac[k] = sum_{i} x[i] * x[k-i] where 0<=i<N and 0<=k-i<N
            # To reduce |ac[k]|, we can try flipping signs of elements that
            # contribute most to ac[k]
            
            # Find which pairs (i, k-i) contribute most
            valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
            if len(valid_i) == 0:
                continue
            
            # Contributions: x[i] * x[k-i] for i in valid_i
            # For each i, the contribution is x[i] * x[k-i]
            # Flipping x[i] changes ac[k] by -2*x[i]*x[k-i]
            
            # Find the i that would give the largest reduction in |ac[k]|
            current_ac_k = ac[k]
            best_reduction = 0
            best_i = None
            
            for i in valid_i:
                j = k - i
                if j < 0 or j >= N:
                    continue
                contrib = best_x[i] * best_x[j]
                # If we flip x[i], new ac[k] = ac[k] - 2*contrib (if i != j)
                # or ac[k] - 2*contrib (if i == j, it's -x[i]^2 so flipping doesn't change)
                if i == j:
                    continue  # flipping doesn't change x[i]^2
                new_ac_k = current_ac_k - 2 * contrib
                if abs(new_ac_k) < abs(current_ac_k) - best_reduction:
                    best_reduction = abs(current_ac_k) - abs(new_ac_k)
                    best_i = i
            
            if best_i is not None and best_reduction > 0:
                # Try flipping best_i
                trial_x = best_x.copy()
                trial_x[best_i] = -trial_x[best_i]
                trial_val = _objective(trial_x)
                if trial_val < best_val:
                    best_x = trial_x
                    best_val = trial_val
                    improved = True
        
        if not improved:
            # Try small random perturbations on elements involved in dominant lags
            perturbation = np.zeros(N)
            for k in top_indices[:5]:
                valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
                if len(valid_i) > 0:
                    idx = valid_i[np.random.randint(len(valid_i))]
                    perturbation[idx] += np.random.normal(0, 0.1)
            
            if np.linalg.norm(perturbation) > 0:
                trial_x = best_x + perturbation
                trial_val = _objective(trial_x)
                if trial_val < best_val:
                    best_x = trial_x
                    best_val = trial_val
                    improved = True
        
        if not improved and it > 100:
            break
    
    return best_x, best_val

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # safety margin before 150s kill
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Structured initializations
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
    
    # Random initializations
    for _ in range(20):
        initializations.append(rng.normal(1.0, 0.5, size=N))
    
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
    for gd_iter in range(30000):
        if not _time_ok():
            break
        
        val_s, grad = _soft_max_and_grad(best_x, tau)
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
    
    # Phase 3: Peak-targeted polish
    if _time_ok():
        polished_x, polished_val = _peak_targeted_polish(best_x, max_iters=5000, time_limit=min(15.0, TIME_LIMIT - (time.time() - START_TIME) - 10.0))
        if polished_val < best_val:
            best_val = polished_val
            best_x = polished_x
    
    # Phase 4: Fine-tuning with small perturbations
    if _time_ok():
        step_size = 0.01
        stall_count = 0
        for iteration in range(200000):
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
