import numpy as np
from scipy.signal import fftconvolve
from scipy.fftpack import dct, idct

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # Initialize coefficients with a Gaussian profile
    # This corresponds to a smooth initial step function
    k = np.arange(N)
    sigma = 50.0
    c = np.exp(-k**2 / (2 * sigma**2))
    c *= 1.0 # Scale
    
    best_s = np.ones(N)
    best_val = np.inf
    
    # Precompute autoconvolution helper
    def compute_autoconv_max(s):
        # s is 0/1 or -1/1
        # Discrete autocorrelation
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return np.max(np.abs(r)) * w, r

    # Initial evaluation
    s_cont = idct(c, type=2, norm='ortho')
    s_sign = np.sign(s_cont)
    s_sign[s_sign == 0] = 1
    val, r = compute_autoconv_max(s_sign)
    if val < best_val:
        best_val = val
        best_s = s_sign.copy()

    for iteration in range(3000):
        # Reconstruct continuous signal from coefficients
        s_cont = idct(c, type=2, norm='ortho')
        
        # Determine sign pattern
        s_sign = np.sign(s_cont)
        s_sign[s_sign == 0] = 1
        
        # Compute objective and subgradient
        val, r = compute_autoconv_max(s_sign)
        
        if val < best_val:
            best_val = val
            best_s = s_sign.copy()
            
        # Find the k that achieves the maximum
        k_max = np.argmax(np.abs(r))
        r_max = r[k_max]
        
        # Subgradient of |r[k_max]| w.r.t s
        # r[k_max] = sum_{i=0}^{N-1-k_max} s[i] * s[i+k_max]
        # d|r[k_max]|/ds[i] = sign(r_max) * (s[i+k_max] (if i < N-k_max) + s[i-k_max] (if i >= k_max))
        sg_s = np.zeros(N)
        sign_r = np.sign(r_max)
        
        # Vectorized gradient computation
        # Term 1: s[i+k_max] for i in [0, N-1-k_max]
        if k_max > 0:
            sg_s[:N-k_max] += s_sign[k_max:]
            sg_s[k_max:] += s_sign[:N-k_max]
        else:
            # k_max == 0, r[0] = sum s[i]^2 = N. Gradient is 2*s[i]
            sg_s[:] = 2.0 * s_sign
            
        sg_s *= sign_r
        
        # Transform gradient to coefficient domain (DCT)
        # Since DCT is orthogonal, the gradient w.r.t c is DCT(gradient w.r.t s)
        # Note: s_cont = IDCT(c). ds/dc = IDCT_matrix. 
        # Gradient w.r.t c = IDCT_matrix^T @ Gradient_w.r.t_s
        # For orthogonal DCT, IDCT_matrix^T = DCT_matrix (with appropriate scaling)
        # scipy.fftpack.dct with norm='ortho' is the inverse of idct with norm='ortho'
        sg_c = dct(sg_s, type=2, norm='ortho')
        
        # Regularization towards initial Gaussian profile
        # c_init = np.exp(-k**2 / (2 * sigma**2))
        reg = 0.01 * (c - np.exp(-k**2 / (2 * sigma**2)))
        
        # Update coefficients
        lr = 0.5 / (1.0 + iteration * 0.01)
        c = c - lr * (sg_c + reg)
        
        # Optional: Clip coefficients to prevent explosion
        # c = np.clip(c, -10, 10)

    # Final reconstruction
    s_cont = idct(c, type=2, norm='ortho')
    s_final = np.sign(s_cont)
    s_final[s_final == 0] = 1
    
    # Check if final is better than best tracked
    val, _ = compute_autoconv_max(s_final)
    if val < best_val:
        return s_final
    else:
        return best_s
