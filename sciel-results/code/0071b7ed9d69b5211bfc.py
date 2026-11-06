import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 115.0

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_value_and_grad(h, beta=100.0):
    """
    Compute smooth value and gradient using log-sum-exp approximation of max.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    lse = m + np.log(sum_exp) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    
    # Gradient calculation
    # val = 2N * lse / s^2 (assuming positive, which it is for our purposes)
    # d(val)/dh = (2N / s^2) * (d(lse)/dh - 2 * lse / s * d(s)/dh)
    
    # p_k = exp(beta * (conv_k - m)) / sum(exp(...))
    p = exp_scaled / sum_exp
    
    # d(lse)/dh_j = sum_k p_k * d(conv_k)/dh_j
    # d(conv_k)/dh_j = h_{j+k} + h_{j-k} (with appropriate bounds)
    # This is a correlation-like operation.
    
    # Let's compute G_j = sum_k p_k * (h_{j+k} + h_{j-k})
    # Using FFT:
    # G = ifft(fft(p) * fft(h)) + ifft(fft(p_rev) * fft(h)) ?
    # Actually, let's just do it directly if N is small enough, or use FFT.
    # N=400, conv size 799. Direct is 400*799 ~ 320k ops, very fast.
    
    # But wait, we need to be careful with indices.
    # conv[k] = sum_i h[i] h[i+k]
    # d(conv[k])/dh[j] = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
    
    # Let's construct the gradient vector directly.
    grad_lse = np.zeros(N)
    for j in range(N):
        # Term 1: h[j+k] for k such that j+k < N => k < N-j
        # Term 2: h[j-k] for k such that j-k >= 0 => k <= j
        # So we sum p_k * h[j+k] for k in [0, N-1-j]
        # and p_k * h[j-k] for k in [0, j]
        
        # This is O(N^2) total, which is 160k operations. Very fast.
        # Let's use vectorized ops for speed.
        
        # Actually, let's just loop, it's simple and robust.
        # Or use np.dot with slices.
        
        # g1 = sum_{k=0}^{N-1-j} p[k] * h[j+k]
        # g2 = sum_{k=0}^{j} p[k] * h[j-k]
        
        if N - 1 - j >= 0:
            g1 = np.dot(p[:N-j], h[j:])
        else:
            g1 = 0.0
        
        if j >= 0:
            g2 = np.dot(p[:j+1], h[j::-1])
        else:
            g2 = 0.0
            
        grad_lse[j] = g1 + g2
        
    # d(s)/dh = 1 for all j
    d_s = np.ones(N)
    
    grad_val = (2.0 * N / (s * s)) * (grad_lse - 2.0 * lse / s * d_s)
    
    # If val was negative (impossible here since abs), sign would flip
    # But lse > 0 always, so val > 0
    
    return val, grad_val

def construct():
    global _t0
    _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Try multiple random initializations
    for restart in range(5):
        if not deadline_ok():
            break
            
        # Initialize with random structure
        # Mix of low frequency and high frequency components
        freqs = rng.uniform(0.1, 5.0, 10)
        h_init = np.zeros(N)
        for f in freqs:
            phase = rng.uniform(0, 2 * np.pi)
            h_init += np.sin(2 * np.pi * f * np.arange(N) / N + phase)
        
        # Normalize to have sum 1
        s = np.sum(h_init)
        if s != 0:
            h_init = h_init / s
        else:
            h_init = np.ones(N) / N
            
        # Add small noise
        h_init += rng.normal(0, 0.01, N)
        
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
        
        v_init = _value(h_init)
        if v_init < best_v:
            best_h = h_init.copy()
            best_v = v_init
        
        # Use L-BFGS-B with smooth approximation
        try:
            result = minimize(
                _smooth_value_and_grad,
                h_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.fun < best_v:
                h_opt = result.x
                v_opt = _value(h_opt)
                if v_opt < best_v and np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                    best_h = h_opt.copy()
                    best_v = v_opt
        except Exception:
            pass
    
    # Local search refinement
    if deadline_ok():
        current_h = best_h.copy()
        current_v = best_v
        lr = 0.01
        
        for step in range(200):
            if not deadline_ok():
                break
            
            noise = rng.normal(0, lr, N)
            trial_h = current_h + noise
            trial_v = _value(trial_h)
            
            if trial_v < current_v:
                current_h = trial_h
                current_v = trial_v
                if current_v < best_v:
                    best_h = current_h.copy()
                    best_v = current_v
                lr *= 0.98
            else:
                lr *= 0.95
                
            if lr < 1e-8:
                lr = 1e-8
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    return best_h
