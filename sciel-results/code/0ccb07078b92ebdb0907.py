"""Combined program: chirp-based initialization with iterative reweighting and soft-max gradient optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.lib.stride_tricks import as_strided

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv

def _hard_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj_and_grad(x, beta=100.0):
    """
    Compute soft-max objective and gradient.
    Objective: 2*N*softmax(conv) / (sum(h))^2
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    # Compute logsumexp with numerical stability
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum_exp = m + np.log(np.sum(exp_shifted)) / beta
    
    # Value: 2*N*log_sum_exp / s^2
    val = 2.0 * N * log_sum_exp / (s * s)
    
    # Gradient computation
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Compute gradient of log_sum_exp w.r.t. h
    # grad_logsumexp[j] = sum_i h[i] * weights[j+i] for j=0,...,N-1
    w_sliding = as_strided(weights, shape=(N, N), strides=(weights.strides[0], weights.strides[0]))
    grad_logsumexp = w_sliding @ h
    
    # d(val)/d(h[j]) = 2*N/s^2 * grad_logsumexp[j] + 2*N*log_sum_exp * (-2/s^3)
    grad_val = (2.0 * N / (s * s)) * (grad_logsumexp - 2.0 * log_sum_exp / s)
    
    return val, grad_val

def _iterative_reweight(h, max_iter=100, tol=1e-10):
    """
    Iterative reweighting: alternate between minimizing max autocorrelation
    and adjusting to equalize the peak.
    """
    h = h.copy().astype(float)
    best_val = _hard_value(h)
    
    for it in range(max_iter):
        s = np.sum(h)
        if abs(s) < 1e-12:
            break
        
        conv = _autoconv_fft(h)
        m = np.max(conv)
        
        # Find the lag(s) achieving max
        jstar = int(np.argmax(conv))
        
        # The gradient direction: we want to reduce c[jstar]
        # d c[jstar]/d h[i] = 2 h[jstar - i] for valid i
        idx = jstar - np.arange(N)
        valid = (idx >= 0) & (idx < N)
        
        # Simple gradient step to reduce the peak
        step = np.zeros(N)
        step[valid] = 2.0 * h[idx[valid]]
        
        # Also account for the denominator change
        # d(1/s^2)/d(h[i]) = -2/s^3
        step -= (-2.0 * m / (s ** 3)) * 2.0 * N / (s * s)
        
        # Normalize step to have unit norm
        snorm = np.linalg.norm(step)
        if snorm < 1e-15:
            break
        step = step / snorm
        
        # Line search
        best_alpha = 0.0
        best_val_step = best_val
        
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            h_trial = h - alpha * step
            s_trial = np.sum(h_trial)
            if abs(s_trial) < 1e-12:
                continue
            c_trial = _autoconv_fft(h_trial)
            m_trial = np.max(c_trial)
            val_trial = abs(2.0 * N * m_trial / (s_trial * s_trial))
            if val_trial < best_val_step:
                best_val_step = val_trial
                best_alpha = alpha
        
        if best_alpha == 0.0:
            break
        
        h = h - best_alpha * step
        best_val = best_val_step
        
        if it > 0 and abs(best_val - best_val_step) < tol:
            break
    
    return h, best_val

def _chirp_init(theta, a=1.0, b=0.5):
    """
    Generate a chirp-like initialization: h[i] = a + b * cos(theta * i)
    This exploits the fact that chirps have bounded autocorrelation peaks.
    """
    i = np.arange(N)
    h = a + b * np.cos(theta * i)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _hard_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # List of initializations
    initializations = []
    
    # 1. Chirp-based initializations (structural innovation)
    for theta in np.linspace(0.01, 2.0 * np.pi, 20):
        for a in [0.5, 1.0, 1.5]:
            for b in [0.1, 0.3, 0.5, 0.8]:
                h0 = _chirp_init(theta, a, b)
                if abs(np.sum(h0)) > 1e-3:
                    initializations.append(h0)
    
    # 2. Random normal with small mean
    for _ in range(10):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Barker-like / Low autocorrelation sequences
    for _ in range(10):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.1 * rng.standard_normal()
        initializations.append(h0)
        
    # 4. Sine waves with random phase
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.5
        initializations.append(h0)

    # 5. Polynomial perturbations of 1s
    for _ in range(5):
        h0 = np.ones(N)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # Optimization parameters - use multiple betas
    betas = [50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        h_curr = h0.copy()
        
        # First apply iterative reweighting
        h_rw, val_rw = _iterative_reweight(h_curr, max_iter=50)
        if val_rw < best_val:
            best_val = val_rw
            best_h = h_rw.copy()
            h_curr = h_rw.copy()
        
        # Then apply soft-max gradient optimization
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            
            try:
                res = minimize(_soft_max_obj_and_grad, h_curr, 
                              jac=lambda x, b=beta: _soft_max_obj_and_grad(x, b)[1],
                              method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _hard_value(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _hard_value(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta on the best solution
    if time.time() < deadline - 5.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(_soft_max_obj_and_grad, best_h,
                              jac=lambda x, b=beta: _soft_max_obj_and_grad(x, b)[1],
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                val = _hard_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Apply iterative reweighting to the best solution
    if time.time() < deadline - 2.0:
        h_final, val_final = _iterative_reweight(best_h, max_iter=100)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
        
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard_value(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
