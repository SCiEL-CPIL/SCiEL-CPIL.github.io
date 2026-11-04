import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE_SECONDS = 115.0

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    """Off-diagonal metric: max |c_k| for k!=0 / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _objective_softmax(x, beta=30.0, include_zero=False):
    """Smooth surrogate: soft-max of |c_k| normalized by S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if include_zero:
        ac = np.abs(c)
    else:
        ac = np.abs(c[1:])
    if len(ac) == 0:
        return 0.0
    m = np.max(ac)
    if m < 1e-15:
        return 0.0
    # Soft max: log(sum(exp(beta*(ac-m))))/beta + m
    # For numerical stability, use the log-sum-exp trick
    shifted = beta * (ac - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    return soft_max / (s * s)

def _grad_objective_softmax(x, beta=30.0, include_zero=False):
    """Gradient of the soft-max objective."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x * 0, 1e10
    c = _autoconv_lin(x)
    if include_zero:
        ac = np.abs(c)
        start = 0
    else:
        ac = np.abs(c[1:])
        start = 1
    
    if len(ac) == 0:
        return x * 0, 0.0
    
    m = np.max(ac)
    if m < 1e-15:
        return x * 0, 0.0
    
    shifted = beta * (ac - m)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # These are the softmax weights
    
    soft_max_val = m + np.log(sum_exp) / beta
    obj = soft_max_val / (s * s)
    
    # Gradient with respect to x
    # d(soft_max)/dx_i = sum_k weights_k * sign(c_k) * d(c_k)/dx_i
    # d(obj)/dx_i = d(soft_max)/dx_i / S^2 + soft_max * d(1/S^2)/dx_i
    # d(1/S^2)/dx_i = -2/S^3
    # So d(obj)/dx_i = (1/S^2) * d(soft_max)/dx_i - 2*soft_max/S^3
    
    n = len(x)
    g_softmax = np.zeros(n)
    
    for idx, k in enumerate(range(start, len(c))):
        wk = weights[idx]
        if wk < 1e-8:
            continue
        sign_ck = np.sign(c[k])
        if k == 0:
            g_softmax += wk * sign_ck * 2 * x
        else:
            g_softmax[:n - k] += wk * sign_ck * x[k:]
            g_softmax[k:] += wk * sign_ck * x[:n - k]
    
    # Chain rule for the 1/S^2 factor
    g_obj = g_softmax / (s * s) - 2 * soft_max_val / (s * s * s) * np.ones(n)
    
    return g_obj, obj

def _project_sum_one(x):
    """Project x to have sum 1."""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return x
    return x - (s - 1.0) / len(x)

def _refine_smooth(x0, deadline, beta_start=30.0, beta_end=100.0, iters=3000, include_zero=False):
    """Refine using gradient descent on the smooth soft-max objective."""
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x = x + 1e-9
        s = np.sum(x)
    x = x / s
    
    best_val = _metric_std(x) if include_zero else _metric_offdiag(x)
    bestx = x.copy()
    
    lr = 1e-3
    no_improve = 0
    patience = 200
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Anneal beta
        progress = it / max(1, iters)
        beta = beta_start + (beta_end - beta_start) * progress
        
        g, obj = _grad_objective_softmax(x, beta=beta, include_zero=include_zero)
        
        # Gradient step
        x_new = x - lr * g
        
        # Project to sum=1
        x_new = _project_sum_one(x_new)
        
        # Clip to reasonable range
        x_new = np.clip(x_new, -5.0, 5.0)
        x_new = _project_sum_one(x_new)
        
        # Evaluate actual metric
        v = _metric_std(x_new) if include_zero else _metric_offdiag(x_new)
        
        if v < best_val:
            best_val = v
            bestx = x_new.copy()
            x = x_new
            no_improve = 0
            lr = min(lr * 1.05, 5e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.7
                no_improve = 0
                if lr < 1e-7:
                    break
    
    return bestx, best_val

def _refine_subgrad(x0, deadline, iters=5000, beta=50.0, mode='offdiag'):
    """Projected subgradient minimization of autoconvolution peaks."""
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x = x + 1e-9
        s = np.sum(x)
    x = x / s
    
    best_val = _metric_std(x) if mode == 'std' else _metric_offdiag(x)
    bestx = x.copy()
    lr = 5e-4
    no_improve = 0
    patience = 500
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        c = _autoconv_lin(x)
        
        if mode == 'offdiag':
            if len(c) <= 1:
                break
            abs_c = np.abs(c[1:])
            if np.max(abs_c) < 1e-15:
                break
            m = np.max(abs_c)
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            
            g = np.zeros(n)
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-6:
                    continue
                sign_ck = np.sign(c[k])
                g[:n - k] += wk * sign_ck * x[k:]
                g[k:] += wk * sign_ck * x[:n - k]
        else:
            abs_c = np.abs(c)
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            
            g = np.zeros(n)
            for k in range(n):
                wk = weights[k]
                if wk < 1e-6:
                    continue
                sign_ck = np.sign(c[k])
                if k == 0:
                    g += wk * sign_ck * 2 * x
                else:
                    g[:n - k] += wk * sign_ck * x[k:]
                    g[k:] += wk * sign_ck * x[:n - k]
        
        x = x - lr * g
        x = _project_sum_one(x)
        x = np.clip(x, -2.0, 2.0)
        x = _project_sum_one(x)
        
        v = _metric_std(x) if mode == 'std' else _metric_offdiag(x)
        if v < best_val:
            best_val = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.02, 2e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.85
                no_improve = 0
                if lr < 1e-6:
                    break
                    
    return bestx, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val_std = _metric_std(best_x)
    
    # Generate diverse initializations
    candidates = []
    
    # 1. Random signs
    for _ in range(10):
        x_rand = rng.choice([-1.0, 1.0], size=N).astype(float)
        s = np.sum(x_rand)
        if abs(s) > 1e-6:
            x_rand = x_rand / s
        candidates.append(x_rand)
    
    # 2. Low frequency sinusoids
    for freq in [1, 2, 3, 4, 5, 7, 8, 11, 13]:
        t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        x_sin = np.sign(np.sin(t))
        s = np.sum(x_sin)
        if abs(s) > 1e-6:
            x_sin = x_sin / s
        candidates.append(x_sin)
    
    # 3. Random smooth functions
    for _ in range(5):
        freqs = rng.integers(1, 10, size=5)
        phases = rng.uniform(0, 2 * np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x_smooth = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            x_smooth += a * np.sin(f * t + p)
        x_smooth = np.sign(x_smooth)
        s = np.sum(x_smooth)
        if abs(s) > 1e-6:
            x_smooth = x_smooth / s
        candidates.append(x_smooth)
    
    # 4. M-sequences
    def _generate_m_sequence(n, poly, init):
        seq = []
        state = init
        for _ in range(n):
            seq.append(1.0 if (state & 1) == 0 else -1.0)
            feedback = 0
            for tap in poly:
                feedback ^= (state >> tap) & 1
            state = (state >> 1) | (feedback << (len(poly) - 1))
        return np.array(seq[:n])
    
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline - 30:
            break
        for init in range(min(3, (1 << len(poly)) - 1)):
            seq = _generate_m_sequence(N, poly, init + 1)
            s = np.sum(seq)
            if abs(s) > 1e-6:
                seq = seq / s
            candidates.append(seq)
    
    # Refine each candidate using both smooth and subgradient methods
    for x_init in candidates:
        if time.time() > deadline - 20:
            break
        
        # Try smooth refinement (off-diagonal)
        x_opt1, v_opt1 = _refine_smooth(x_init, deadline, iters=1500, include_zero=False)
        v_std1 = _metric_std(x_opt1)
        if v_std1 < best_val_std:
            best_val_std = v_std1
            best_x = x_opt1
        
        # Try subgradient refinement on the result
        if time.time() > deadline - 15:
            continue
        x_opt2, v_opt2 = _refine_subgrad(x_opt1, deadline, iters=2000, mode='offdiag')
        v_std2 = _metric_std(x_opt2)
        if v_std2 < best_val_std:
            best_val_std = v_std2
            best_x = x_opt2
    
    # Final polish on the best candidate
    if time.time() < deadline - 10:
        x_final, v_final = _refine_smooth(best_x, deadline, iters=3000, beta_start=50.0, beta_end=200.0, include_zero=False)
        v_std_final = _metric_std(x_final)
        if v_std_final < best_val_std:
            best_val_std = v_std_final
            best_x = x_final
        
        if time.time() < deadline - 5:
            x_final2, v_final2 = _refine_subgrad(best_x, deadline, iters=3000, mode='offdiag', beta=100.0)
            v_std_final2 = _metric_std(x_final2)
            if v_std_final2 < best_val_std:
                best_val_std = v_std_final2
                best_x = x_final2
    
    return best_x
