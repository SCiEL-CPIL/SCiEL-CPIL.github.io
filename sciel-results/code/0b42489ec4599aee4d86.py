import numpy as np
import time

N = 400

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
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, tau=0.1):
    """
    Compute a smooth approximation of max(|ac|)/(sum x)^2 and its gradient.
    Uses soft-max over |ac| values.
    Optimized gradient calculation using FFT.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max approximation of max
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Weighted sum: w_i = exp((abs_ac_i - m)/tau) / sum(exp((abs_ac_j - m)/tau))
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    # Soft max value
    soft_max = m + tau * np.log(w_sum)
    
    # Objective value
    val = soft_max / (s * s)
    
    # Gradient computation
    # v[i] = weights[i] * sign(ac[i])
    v = np.sign(ac) * weights
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    V = np.fft.rfft(v, n=L_fft)
    X = np.fft.rfft(x, n=L_fft)
    
    # Term 1: sum_k v_k x_{k-j} -> Correlation (v * x) in FFT domain is V * conj(X)
    corr = np.fft.irfft(V * np.conj(X), n=L_fft)
    
    # Term 2: sum_k v_k x_{j-k} -> Convolution (v * x) in FFT domain is V * X
    conv = np.fft.irfft(V * X, n=L_fft)
    
    # grad_soft_max[j] = corr[j] + conv[j]
    grad_sm = corr[:N] + conv[:N]
    
    # Gradient of objective
    # J = soft_max / s^2
    # dJ/dx = (grad_sm * s^2 - soft_max * 2s) / s^4 = grad_sm/s^2 - 2*soft_max/s^3
    grad_obj = grad_sm / (s * s) - 2 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # safety margin before 150s kill

    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    initializations.append(np.ones(N))
    
    t = np.linspace(0, 1, N)
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.05, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
        
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
        
    # Add random perturbations to some
    for i in range(10):
        init = initializations[i % len(initializations)].copy()
        init += rng.normal(0, 0.1, size=N)
        initializations.append(init)

    # Evaluate initializations
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    # Run multiple independent gradient descent runs
    # We will allocate time to each run
    num_runs = 10
    time_per_run = (TIME_LIMIT - (time.time() - START_TIME)) / num_runs
    
    for run_idx in range(num_runs):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Pick a starting point: either the current best or a random one
        if run_idx == 0:
            x = best_x.copy()
        else:
            # Mix of best and random
            if run_idx % 2 == 0:
                x = best_x.copy()
            else:
                x = initializations[run_idx % len(initializations)].copy()
            
            # Add small noise to escape local minima if restarting from best
            if run_idx > 1 and run_idx % 3 == 0:
                x += rng.normal(0, 0.05, size=N)
        
        current_val = _objective(x)
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
            
        tau = 0.05
        lr = 0.1
        max_iter = 2000
        
        for gd_iter in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            if time.time() - START_TIME > (TIME_LIMIT - time_per_run * (num_runs - run_idx - 1)):
                # Time to move to next run
                break
                
            val_s, grad = _smooth_objective_and_grad(x, tau)
            if not np.isfinite(val_s):
                break
                
            actual_val = _objective(x)
            
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-8:
                break
                
            # Backtracking line search
            alpha = 1.0
            improved = False
            for _ in range(20):
                x_trial = x - alpha * lr * grad
                trial_val = _objective(x_trial)
                if trial_val < actual_val:
                    x = x_trial
                    actual_val = trial_val
                    improved = True
                    break
                alpha *= 0.5
                
            if not improved:
                lr *= 0.5
                if lr < 1e-8:
                    break
            else:
                lr = min(0.5, lr * 1.05)
                
            # Anneal tau
            if gd_iter % 100 == 0 and tau > 0.005:
                tau *= 0.8

            # Update best
            if actual_val < best_val:
                best_val = actual_val
                best_x = x.copy()
                
    # Final polish: high precision local search on the best solution
    # Use very small steps and tight tolerance
    x = best_x.copy()
    current_val = _objective(x)
    tau = 0.001
    lr = 0.01
    
    for gd_iter in range(5000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        val_s, grad = _smooth_objective_and_grad(x, tau)
        if not np.isfinite(val_s):
            break
            
        actual_val = _objective(x)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-10:
            break
            
        alpha = 1.0
        improved = False
        for _ in range(10):
            x_trial = x - alpha * lr * grad
            trial_val = _objective(x_trial)
            if trial_val < actual_val:
                x = x_trial
                actual_val = trial_val
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            lr *= 0.5
            if lr < 1e-10:
                break
        else:
            lr = min(0.1, lr * 1.1)
            
        if actual_val < best_val:
            best_val = actual_val
            best_x = x.copy()

    return best_x
