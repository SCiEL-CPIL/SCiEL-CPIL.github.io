import numpy as np
from scipy.optimize import minimize

N = 50

def _ratio(h):
    # Normalize to L1=1
    s = np.sum(h)
    if s == 0: return 0.0
    h = h / s
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    cc = np.fft.irfft(H * H.conj(), n=n_fft)
    peak = np.max(cc[:N])
    if peak == 0: return 0.0
    return l2_sq / peak

def construct():
    # Start with a Gaussian-like initial guess
    x = np.linspace(-1, 1, N)
    h0 = np.exp(-x**2 * 5)
    
    # Optimize h to maximize ratio
    # We maximize log(ratio) or just ratio
    # Since ratio is scale invariant, we can fix sum(h)=1 in the function
    
    best_h = h0
    best_val = _ratio(h0)
    
    # Simple random restart / perturbation search
    # Given 120s limit, we can do many iterations
    for _ in range(5000):
        h_try = best_h + np.random.normal(0, 0.1, N)
        h_try = np.maximum(h_try, 0) # Nonnegative constraint
        if np.sum(h_try) == 0: continue
        val = _ratio(h_try)
        if val > best_val:
            best_val = val
            best_h = h_try.copy()
            
    return list(best_h)
