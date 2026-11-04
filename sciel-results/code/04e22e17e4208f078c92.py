import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def smooth_objective_and_grad(h, beta=50.0):
    """
    Compute smooth approximation of the score using log-sum-exp, plus gradient.
    Score = 0.5 * max(autoconv) / integral^2
    Smooth: 0.5 * (1/beta) * log(sum(exp(beta * autoconv))) / integral^2
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 1e10, np.zeros_like(h)
    
    autoconv = compute_autoconv(h)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return 1e10, np.zeros_like(h)
    
    # Log-sum-exp for stability
    max_ac = np.max(autoconv)
    exp_ac = np.exp(beta * (autoconv - max_ac))
    log_sum_exp = max_ac + np.log(np.sum(exp_ac)) / beta
    
    score = 0.5 * log_sum_exp / (integral ** 2)
    
    # Gradient with respect to h
    # d(autoconv)/d(h) via FFT
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    w = exp_ac / np.sum(exp_ac)  # weights, sum to 1
    
    # Efficient gradient computation using FFT
    # autoconv_k = DELTA * sum_j h_j * h_{k-j}
    # d(autoconv_k)/d(h_j) = DELTA * (h_{k-j} + h_{j-k}) for valid indices
    # 
    # We need: grad_h[j] = DELTA * sum_k w_k * (h[k-j] + h[j-k])
    # 
    # Term 1: sum_k w_k * h[k-j] = (w * h)[j] where * is cross-correlation
    # Term 2: sum_k w_k * h[j-k] = (w * h)[j] where * is convolution
    # 
    # Using FFT:
    # For cross-correlation: (w * h)[j] = ifft(FFT(w) * conj(FFT(h)))
    # For convolution: (w * h)[j] = ifft(FFT(w) * FFT(h))
    # But we need to handle the bounds properly.
    
    # Pad w and h to length fft_size
    w_padded = np.zeros(fft_size)
    w_padded[:size] = w
    
    h_padded = np.zeros(fft_size)
    h_padded[:N] = h
    
    W = np.fft.rfft(w_padded)
    H = np.fft.rfft(h_padded)
    
    # Term 1: sum_k w_k * h[k-j]
    # This is cross-correlation: (w ⋆ h)[j] = sum_k w[k] * h[k-j]
    # In frequency domain: ifft(conj(FFT(w)) * FFT(h))
    # But we need to be careful with the indexing.
    # Actually, let's think about it differently.
    # 
    # Let's define: for j in [0, N-1],
    # term1[j] = sum_{k=0}^{2N-2} w[k] * h[k-j] where h[m]=0 if m not in [0,N-1]
    # = sum_{m=0}^{N-1} w[m+j] * h[m]  (since k=m+j, and k must be in [0,2N-2])
    # 
    # This is like a sliding dot product. We can compute it via:
    # term1 = np.array([np.dot(w[j:j+N], h) for j in range(N)])
    # But that's O(N^2). Let's use FFT.
    # 
    # term1[j] = sum_{m=0}^{N-1} h[m] * w[j+m]
    # This is the (j-th) element of the cross-correlation of h and w, but only for j in [0,N-1]
    # 
    # Cross-correlation: (h ⋆ w)[j] = sum_m h[m] * w[m+j]
    # In frequency: ifft(FFT(h) * conj(FFT(w)))
    
    # Actually, let me just use a direct but vectorized approach since N=600 is small
    # term1[j] = sum_{m=0}^{N-1} h[m] * w[j+m] for j in [0, N-1]
    # term2[j] = sum_{m=0}^{N-1} h[m] * w[j-m] for j in [0, N-1] (with w[k]=0 if k<0 or k>=2N-1)
    
    # For term1: create a matrix where row j is w[j:j+N]
    # But that's N x N = 360K elements, which is fine
    
    # More efficient: use sliding window via FFT
    # term1[j] = sum_{m=0}^{N-1} h[m] * w[j+m]
    # Let's define w_ext of length 2N-1 (already have w)
    # term1[j] = dot(h, w[j:j+N])
    
    # Using FFT-based approach:
    # Pad h to length 2N-1 (zero-pad the rest)
    h_ext = np.zeros(size)
    h_ext[:N] = h
    
    # term1[j] = sum_m h_ext[m] * w[j+m] for m in [0, N-1]
    # This is the cross-correlation of h_ext and w, evaluated at lags j
    # (h_ext ⋆ w)[j] = sum_m h_ext[m] * w[m+j]
    
    # Compute via FFT:
    H_ext = np.fft.rfft(h_ext, n=fft_size)
    W_full = np.fft.rfft(w_padded)
    
    # Cross-correlation: ifft(conj(FFT(h_ext)) * FFT(w))
    # Actually: (a ⋆ b)[n] = sum_k a[k] * b[k+n]
    # FFT: ifft(conj(FFT(a)) * FFT(b)) gives (a ⋆ b)
    
    corr1 = np.fft.irfft(np.conj(H_ext) * W_full, n=fft_size)
    # corr1[j] = sum_k h_ext[k] * w[k+j] for j in [0, fft_size-1]
    # We want j in [0, N-1]
    term1 = corr1[:N]
    
    # term2[j] = sum_{k=0}^{2N-2} w[k] * h[j-k]
    # = sum_{m=0}^{N-1} w[j-m] * h[m]  (let m=j-k, k=j-m)
    # = sum_{m=0}^{N-1} h[m] * w[j-m]
    # This is the cross-correlation of w and h: (w ⋆ h)[j] = sum_m w[m] * h[m-j]
    # Wait, let me redefine.
    # term2[j] = sum_{m=0}^{N-1} h[m] * w[j-m]
    # Let n = j-m, then m = j-n
    # = sum_{n=j-(N-1)}^{j} h[j-n] * w[n]
    # = sum_{n=0}^{j} h[j-n] * w[n]  (since h[m]=0 for m<0)
    # This is the convolution of h and w, evaluated at j
    # (h * w)[j] = sum_n h[n] * w[j-n]
    
    # Convolution via FFT: ifft(FFT(h) * FFT(w))
    conv2 = np.fft.irfft(H_ext * W_full, n=fft_size)
    # conv2[j] = sum_n h_ext[n] * w[j-n] for j in [0, fft_size-1]
    # We want j in [0, N-1]
    term2 = conv2[:N]
    
    grad_h = DELTA * (term1 + term2)
    
    d_lse_dh = grad_h
    
    d_int_dh = DELTA * np.ones(N)
    
    d_score_dh = 0.5 * (d_lse_dh / (integral ** 2) - 2 * log_sum_exp / (integral ** 3) * d_int_dh)
    
    return score, d_score_dh

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Start with uniform
    h = np.ones(N)
    best_score = compute_score(h)
    best_h = h.copy()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N  # midpoints
    
    # 1. Parametric Fourier Optimization with smooth objective
    n_modes = 25
    def param_to_h(params):
        h = 1.0
        for k in range(1, n_modes + 1):
            h += params[2*k-2] * np.cos(2*np.pi*k*x)
            h += params[2*k-1] * np.sin(2*np.pi*k*x)
        return np.maximum(h, 0)
        
    def objective_and_grad_fourier(params):
        h = param_to_h(params)
        score, grad_h = smooth_objective_and_grad(h, beta=50.0)
        
        # Chain rule: d(score)/d(params) = d(score)/d(h) * d(h)/d(params)
        grad_params = np.zeros(2 * n_modes)
        for k in range(1, n_modes + 1):
            grad_params[2*k-2] = np.dot(grad_h, -2*np.pi*k*np.sin(2*np.pi*k*x))
            grad_params[2*k-1] = np.dot(grad_h, 2*np.pi*k*np.cos(2*np.pi*k*x))
        
        return score, grad_params
        
    # Multi-start L-BFGS-B with analytical gradient
    np.random.seed(42)
    best_fourier_score = np.inf
    best_fourier_params = None
    
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        if restart == 0:
            x0 = np.zeros(2 * n_modes)
        else:
            x0 = np.random.randn(2 * n_modes) * 0.2
            
        try:
            res = optimize.minimize(objective_and_grad_fourier, x0, method='L-BFGS-B', 
                                    jac=True,
                                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < best_fourier_score:
                best_fourier_score = res.fun
                best_fourier_params = res.x.copy()
                h_candidate = param_to_h(res.x)
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # 2. Direct Optimization on 600 heights with smooth objective and analytical gradient
    if time.time() < deadline - 20:
        h_start = best_h.copy()
        
        def objective_and_grad_direct(h_vec):
            if np.any(h_vec < 0):
                return 1e10, np.zeros_like(h_vec)
            return smooth_objective_and_grad(h_vec, beta=50.0)
            
        try:
            res_direct = optimize.minimize(objective_and_grad_direct, h_start, method='L-BFGS-B',
                                           jac=True,
                                           bounds=[(0, None)] * N,
                                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
            if res_direct.fun < best_score:
                h_candidate = res_direct.x
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
        except Exception:
            pass
        
    # 3. Final high-precision polish
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        
        def objective_and_grad_direct_final(h_vec):
            if np.any(h_vec < 0):
                return 1e10, np.zeros_like(h_vec)
            return smooth_objective_and_grad(h_vec, beta=100.0)
        
        try:
            res_final = optimize.minimize(objective_and_grad_direct_final, h_start, method='L-BFGS-B',
                                          jac=True,
                                          bounds=[(0, None)] * N,
                                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res_final.fun < best_score:
                h_candidate = res_final.x
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
