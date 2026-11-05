"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT (O(N log N) instead of O(N^2))."""
    n = len(h)
    # Pad to avoid circular convolution effects
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2*n-1]

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj_grad(x, beta=50.0):
    """
    Objective and gradient for: 2*N*softmax(autoconv(h)) / (sum(h))^2
    Using soft-max: m + (1/beta)*log(sum(exp(beta*(conv_i - m))))
    where m = max(conv)
    
    Returns (value, gradient)
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    if m <= 0:
        # If max is non-positive, the abs makes it -m, so we want to maximize m
        # gradient of -m w.r.t. h: -d(max)/dh
        # For simplicity, use soft-max formulation regardless
        pass
    
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = m + np.log(sum_exp) / beta
    
    value = 2.0 * N * soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/d(conv_i) = exp(shifted_i) / sum_exp
    weights = exp_shifted / sum_exp
    
    # d(conv_j)/d(h_k) for full autoconvolution:
    # conv[j] = sum_k h[k] * h[j-k] (with appropriate bounds)
    # d(conv_j)/d(h_k) = h[j-k] + h[k-j] if both indices valid
    
    # For FFT-based approach, we can compute the gradient more efficiently
    # conv = ifft(fft(h)^2)
    # d(conv)/d(h) involves the convolution structure
    
    # Let's compute gradient numerically-free using the structure:
    # conv[j] = sum_{k=max(0,j-N+1)}^{min(j,N-1)} h[k]*h[j-k]
    # d(conv[j])/d(h[k]) = h[j-k] + h[k-j] (when both in range)
    
    # This is equivalent to: grad_conv[j] = h[j] + h[-j] (circular, with care)
    # Actually for full convolution:
    # d(conv[j])/d(h[k]) = h[j-k] if 0<=j-k<N else 0, plus h[k-j] if 0<=k-j<N else 0
    
    # Efficient computation: 
    # grad_h[k] = sum_j weights[j] * d(conv[j])/d(h[k])
    #           = sum_j weights[j] * (h[j-k] + h[k-j])
    # This is a correlation-like operation
    
    # Let's compute it as: grad_h = conv(weights, h) + conv(weights, h[::-1]) ... 
    # Actually simpler: use the fact that the Jacobian of conv w.r.t. h is a structured matrix
    
    # For efficiency, compute via FFT:
    # The gradient of sum_j w_j * conv[j] w.r.t. h[k] 
    # = sum_j w_j * (h[j-k] + h[k-j]) for valid indices
    # = (w * h)[k] + (w * h_flipped)[k] where * denotes cross-correlation
    
    # Cross-correlation of w and h: sum_j w[j] * h[j-k] 
    # This can be computed via FFT
    
    w = weights
    # corr1[k] = sum_j w[j] * h[j-k] (valid range)
    # corr2[k] = sum_j w[j] * h[k-j] (valid range)
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # corr1 = ifft(W * conj(H)) gives sum_j w[j] * h[j+k]... need to be careful
    # Let's just do it directly for correctness, N=400 is small enough
    
    # grad_h[k] = sum_j weights[j] * (h[j-k] if 0<=j-k<N else 0 + h[k-j] if 0<=k-j<N else 0)
    
    grad_h = np.zeros(n)
    for k in range(n):
        # Term 1: sum_j weights[j] * h[j-k] for j-k in [0,N-1]
        # j ranges from k to k+N-1, but j < 2N-1 (length of conv)
        # h[j-k] valid when 0 <= j-k < N, i.e., k <= j < k+N
        j_min = max(0, k)
        j_max = min(2*n-2, k+n-1)
        if j_min <= j_max:
            js = np.arange(j_min, j_max+1)
            idx_h = js - k
            grad_h[k] += np.sum(weights[js] * h[idx_h])
        
        # Term 2: sum_j weights[j] * h[k-j] for k-j in [0,N-1]
        # j ranges from k-(N-1) to k, i.e., k-N+1 <= j <= k
        j_min = max(0, k-n+1)
        j_max = min(2*n-2, k)
        if j_min <= j_max:
            js = np.arange(j_min, j_max+1)
            idx_h = k - js
            grad_h[k] += np.sum(weights[js] * h[idx_h])
    
    # Now compute full gradient of value = 2*N*soft_max / s^2
    # d(value)/d(h[k]) = 2*N * (d(soft_max)/d(h[k]) * s^2 - soft_max * 2*s * d(s)/d(h[k])) / s^4
    #                  = 2*N * (grad_sm[k] / s - 2*soft_max * (1/s^2) * (1/s) * s) 
    # Let me redo: value = 2*N*soft_max * s^{-2}
    # d(value)/d(h[k]) = 2*N * (d(soft_max)/d(h[k]) * s^{-2} + soft_max * (-2) * s^{-3} * 1)
    #                  = 2*N * s^{-3} * (s * grad_sm[k] - 2 * soft_max)
    
    # But we need d(soft_max)/d(h[k]) = sum_j weights[j] * d(conv[j])/d(h[k]) = grad_h[k]
    # Wait, soft_max = m + log(sum_exp)/beta, and weights = exp_shifted/sum_exp
    # d(soft_max)/d(conv[j]) = weights[j] (for the soft-max part)
    # But m also depends on conv... for the soft-max formulation, 
    # d(soft_max)/d(conv[j]) = weights[j] exactly (this is a known property)
    
    # So d(soft_max)/d(h[k]) = sum_j weights[j] * d(conv[j])/d(h[k]) = grad_h[k]
    
    d_soft_max_dh = grad_h
    d_value_dh = 2.0 * N * (s * d_soft_max_dh - 2.0 * soft_max) / (s ** 3)
    
    return value, d_value_dh

def _soft_max_obj(x, beta=50.0):
    """Objective only (for compatibility)."""
    val, _ = _soft_max_obj_grad(x, beta)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(8):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns (alternating signs with varying period)
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20, 40]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Sparse spikes
    for n_spikes in [5, 10, 20]:
        h0 = np.zeros(N)
        positions = rng.choice(N, size=n_spikes, replace=False)
        h0[positions] = rng.standard_normal(n_spikes)
        h0 += 0.01 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. Smooth low-frequency patterns
    for num_modes in [3, 5, 10]:
        h0 = np.zeros(N)
        for k in range(1, num_modes+1):
            h0 += rng.standard_normal() * np.cos(2*np.pi*k*np.arange(N)/N)
            h0 += rng.standard_normal() * np.sin(2*np.pi*k*np.arange(N)/N)
        h0 += 1.0  # baseline to ensure non-zero sum
        initializations.append(h0)
    
    # Optimize each initialization with gradient-based method
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        val_curr = _objective_hard(h_curr)
        
        # Direct optimization with analytic gradient
        if time.time() < deadline - 15.0:
            try:
                def obj_grad_wrapper(x, beta=30.0):
                    return _soft_max_obj_grad(x, beta)
                
                res = minimize(obj_grad_wrapper, h_curr,
                              method='L-BFGS-B',
                              jac=True,
                              args=(30.0,),
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8,
                                       'maxls': 50, 'maxfun': 500})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
        
        # Try different beta values
        if time.time() < deadline - 10.0:
            for beta in [20.0, 50.0, 100.0]:
                if time.time() > deadline - 8.0:
                    break
                try:
                    def obj_grad_wrapper2(x, beta=beta):
                        return _soft_max_obj_grad(x, beta)
                    
                    res = minimize(obj_grad_wrapper2, h_curr,
                                  method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8,
                                           'maxls': 50, 'maxfun': 400})
                    val = _objective_hard(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
                except Exception:
                    pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                def obj_grad_polish(x, beta=beta):
                    return _soft_max_obj_grad(x, beta)
                
                res = minimize(obj_grad_polish, best_h,
                              method='L-BFGS-B',
                              jac=True,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                       'maxls': 100, 'maxfun': 1000})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
