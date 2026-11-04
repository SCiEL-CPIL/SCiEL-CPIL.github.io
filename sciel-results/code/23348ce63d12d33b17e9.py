"""Combined: Zadoff-Chu chirp parameterization + L-BFGS-B smooth off-diagonal optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
# Safety margin before the 150s kill
SAFETY = 10.0

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _chirp_sequence(alpha, beta, gamma, phi0):
    """
    Generate a real-valued sequence from a quadratic chirp phase.
    """
    n = np.arange(N)
    phase = alpha * n**2 + beta * n + gamma + phi0
    x = np.cos(phase)
    return x

def _gradient_descent_polish(x0, max_iter=2000, initial_lr=0.005, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    prev_val = best_val
    stall_count = 0
    max_stall = 50
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        # Soft max gradient
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Compute soft max and its gradient
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        # Gradient of soft_max w.r.t. abs_c
        grad_abs_c = exp_vals / sum_exp
        
        # Gradient w.r.t. c (chain rule: d|c|/dc = sign(c))
        c_off = c[1:]
        sign_c = np.sign(c_off)
        # Handle zeros: use subgradient 0
        sign_c[abs_c < 1e-15] = 0.0
        grad_c = grad_abs_c * sign_c
        
        # Gradient of autoconvolution w.r.t. x
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        # Account for the (sum x)^2 denominator
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        # Gradient of metric
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        # Adaptive step: backtracking line search
        best_x_new = x
        best_val_new = prev_val
        for trial in range(10):
            x_new = x - lr * grad_metric
            s_new = np.sum(x_new)
            if abs(s_new) < 1e-12:
                break
            x_new = x_new / s_new
            val_new = _metric_offdiag(x_new)
            if val_new < prev_val * (1.0 - 1e-4):
                best_x_new = x_new
                best_val_new = val_new
                break
            lr *= 0.5
            if lr < 1e-10:
                break
        
        x = best_x_new
        prev_val = best_val_new
        
        if val_new < best_val:
            best_val = val_new
            best_x = x.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Decay learning rate and temperature
        lr *= 0.998
        temp *= 0.9995
        
        if stall_count > max_stall:
            break
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - SAFETY
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Quick chirp starts (much faster than grid search)
    chirp_candidates = []
    alpha_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    beta_vals = [0.0, 0.1, 0.2, 0.3]
    
    for alpha in alpha_vals:
        if time.time() > deadline - 60:
            break
        for beta in beta_vals:
            if time.time() > deadline - 60:
                break
            # Quick optimization of gamma, phi0
            try:
                res = minimize(
                    lambda p, a=alpha, b=beta: _objective_smooth(_chirp_sequence(a, b, p[0], p[1]), temp=0.5),
                    [0.0, 0.0],
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                x_chirp = _chirp_sequence(alpha, beta, res.x[0], res.x[1])
                chirp_candidates.append(x_chirp)
                val = _metric_offdiag(x_chirp)
                if val < best_val:
                    best_val = val
                    best_x = x_chirp.copy()
            except Exception:
                pass
    
    # Strategy 2: Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Random sequences with non-zero mean
    for _ in range(20):
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
            
    # 3. Pseudo-random binary
    for _ in range(15):
        x = rng.choice([-1, 1], N)
        if abs(np.sum(x)) > 5:
            candidates.append(x)

    # 4. Structured: Block patterns
    for block_size in [5, 10, 20, 50, 100]:
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # 5. Sine wave patterns
    for freq in [1, 2, 5, 10]:
        for phase in [0, np.pi/2, np.pi]:
            t = np.arange(N)
            pattern = np.sin(2 * np.pi * freq * t / N + phase)
            pattern = np.sign(pattern)
            if abs(np.sum(pattern)) < 1:
                pattern[0] = 1.0
            candidates.append(pattern)
    
    # 6. Add chirp candidates
    for xc in chirp_candidates:
        candidates.append(xc.copy())
    
    # Add the best so far
    candidates.append(best_x.copy())
    
    # Optimize candidates with L-BFGS-B
    for x0 in candidates:
        if time.time() > deadline - 30:
            break
            
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # First pass with higher temperature
            res1 = minimize(
                lambda x, t=1.0: _objective_smooth(x, temp=t),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_mid = res1.x
            
            # Second pass with lower temperature
            res2 = minimize(
                lambda x, t=0.1: _objective_smooth(x, temp=t),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass

    # Final polish with gradient descent on the best candidate
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x, 
                max_iter=3000, 
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass

    return best_x
