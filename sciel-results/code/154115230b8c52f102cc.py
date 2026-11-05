import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autoconv_max_soft(h, beta=50.0):
    """Soft-max of autoconvolution with analytic gradient support."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        soft_max = m + np.log(np.sum(exp_shifted)) / beta
        return soft_max, weights
    else:
        return m, np.zeros_like(conv)

def _objective_soft(h, beta=50.0):
    """Objective: 2*N*soft_max(autoconv) / (sum(h))^2"""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    soft_max, _ = _autoconv_max_soft(h, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    """Analytic gradient of soft objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return np.zeros_like(h)
    
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    
    # Gradient of soft_max w.r.t. h[i]:
    # d(soft_max)/d(h_i) = sum_j weights[j] * d(conv[j])/d(h_i)
    # conv[j] = sum_k h[k] * h[j-k] (where 0<=k<N, 0<=j-k<N)
    # d(conv[j])/d(h_i) = h[j-i] + h[i-j] (with boundary checks)
    # 
    # Vectorized: grad_soft[i] = sum_j weights[j] * (h[j-i] if 0<=j-i<N else 0)
    #                              + sum_j weights[j] * (h[i-j] if 0<=i-j<N else 0)
    # The second sum is just the first with j->N-1-j symmetry, but let's compute both.
    
    # For efficiency, compute via correlation-like operations
    # grad_soft[i] = sum_j weights[j] * h[j-i] + sum_j weights[j] * h[i-j]
    # First term: for each i, sum over j where 0<=j-i<N, i.e., i<=j<=i+N-1
    # This is a sliding window sum of weights * h
    
    # Use FFT for speed: grad_soft[i] = (weights * h)[i] in circular convolution sense
    # But we need linear convolution with proper bounds
    
    # Simpler: direct computation is O(N^2) = 160000, which is fine
    grad_soft = np.zeros(N)
    for i in range(N):
        # sum_j weights[j] * h[j-i] where 0<=j-i<N
        # j ranges from max(0,i) to min(2N-2, i+N-1)
        j_min = max(0, i)
        j_max = min(2*N-2, i+N-1)
        if j_min <= j_max:
            indices = np.arange(j_min, j_max+1)
            h_indices = indices - i
            grad_soft[i] += np.sum(weights[j_min:j_max+1] * h[h_indices])
        
        # sum_j weights[j] * h[i-j] where 0<=i-j<N
        # j ranges from max(0, i-N+1) to min(2N-2, i)
        j_min2 = max(0, i-N+1)
        j_max2 = min(2*N-2, i)
        if j_min2 <= j_max2:
            indices2 = np.arange(j_min2, j_max2+1)
            h_indices2 = i - indices2
            grad_soft[i] += np.sum(weights[j_min2:j_max2+1] * h[h_indices2])
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_soft_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_soft_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_soft * s - 2.0 * soft_max) / (s * s * s)

def _objective_hard(h):
    """Exact objective for evaluation."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct_flatten_obj(h, lam=0.1):
    """Combined objective: peak minimization + spectral flattening."""
    # DCT-II
    from scipy.fft import dct
    dct_coeffs = dct(h, type=2, norm='ortho')
    # Penalize large coefficients (flat spectrum)
    flat_penalty = lam * np.sum(np.abs(dct_coeffs)**4) / N
    peak_val = _objective_soft(h, beta=50.0)
    return peak_val + flat_penalty

def _dct_flatten_grad(h, lam=0.1):
    """Gradient of combined objective."""
    peak_grad = _objective_soft_grad(h, beta=50.0)
    
    from scipy.fft import dct, idct
    dct_coeffs = dct(h, type=2, norm='ortho')
    # d/dh_i sum_j |dct_j|^4 = sum_j 4*|dct_j|^2 * sign(dct_j) * d(dct_j)/d(h_i)
    # For DCT-II with ortho norm, d(dct_j)/d(h_i) = sqrt(2/N)*c_j*cos(pi*j*(i+0.5)/N)
    # This is complex; use numerical for the DCT part
    eps = 1e-6
    dct_grad = np.zeros(N)
    for i in range(N):
        hp = h.copy()
        hp[i] += eps
        hm = h.copy()
        hm[i] -= eps
        dp = dct(hp, type=2, norm='ortho')
        dm = dct(hm, type=2, norm='ortho')
        dct_grad[i] = (np.sum(np.abs(dp)**4) - np.sum(np.abs(dm)**4)) / (2*eps)
    
    return peak_grad + lam * dct_grad / N

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    i = np.arange(N)
    rng = np.random.default_rng(12345)
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Low-frequency sinusoids with DC
    for k in [1, 2, 3]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5, 8]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [5, 10, 20, 50]:
        h0 = np.sin(2*np.pi*freq*i/N) + 0.3*np.cos(4*np.pi*freq*i/N)
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Block patterns
    for block_size in [10, 20, 40, 80]:
        n_blocks = N // block_size
        h0 = np.zeros(N)
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 5. Random with structure
    for _ in range(3):
        h0 = rng.standard_normal(N)
        # Low-pass filter
        from scipy.signal import butter, filtfilt
        try:
            b, a = butter(3, 0.3, btype='low')
            h0 = filtfilt(b, a, h0)
        except:
            pass
        initializations.append(h0)
    
    # Optimize each initialization
    betas = [30.0, 50.0, 80.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
            
            # Phase 1: Pure peak minimization
            try:
                res = minimize(_objective_soft, h_curr,
                             method='L-BFGS-B',
                             args=(beta,),
                             jac=_objective_soft_grad,
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7})
                h_curr = res.x
            except Exception:
                pass
            
            # Phase 2: Spectral flattening refinement
            if time.time() < deadline - 15.0:
                try:
                    res2 = minimize(_dct_flatten_obj, h_curr,
                                  method='L-BFGS-B',
                                  args=(0.05,),
                                  jac=_dct_flatten_grad,
                                  options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7})
                    h_curr = res2.x
                except Exception:
                    pass
            
            # Phase 3: Final peak polishing
            if time.time() < deadline - 10.0:
                try:
                    res3 = minimize(_objective_soft, h_curr,
                                  method='L-BFGS-B',
                                  args=(80.0,),
                                  jac=_objective_soft_grad,
                                  options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-9})
                    h_curr = res3.x
                except Exception:
                    pass
            
            # Evaluate
            val = _objective_hard(h_curr)
            if val < best_val:
                best_val = val
                best_h = h_curr.copy()
    
    # Final verification
    if _objective_hard(best_h) >= best_val - 1e-6:
        return best_h
    else:
        return np.ones(N)
