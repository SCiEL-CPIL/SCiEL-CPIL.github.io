import numpy as np
import time
from scipy.optimize import minimize
import math

N = 600

def compute_ac(f):
    # Linear autocorrelation
    nfft = 2 * N
    F = np.fft.rfft(f, n=nfft)
    ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
    return ac

def objective_and_grad(f, beta):
    # f is vector
    s = np.sum(f)
    if s < 1e-10:
        return 1e10, np.zeros_like(f)
    
    ac = compute_ac(f)
    # LSE
    max_ac = np.max(ac)
    # Shift for stability
    ac_shifted = ac - max_ac
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = np.sum(exp_ac)
    lse = max_ac + (1.0/beta) * np.log(sum_exp)
    
    # Objective value
    val = lse / (s * s)
    
    # Gradient
    # d/d f_i [ lse / s^2 ]
    # = ( s^2 * d(lse)/df_i - lse * 2 s ) / s^4
    # = ( s * d(lse)/df_i - 2 lse ) / s^3
    
    # d(lse)/df_i = sum_k w_k * d(ac_k)/df_i
    # where w_k = exp(beta * ac_k) / sum_exp
    # d(ac_k)/df_i = f[i+k] + f[i-k] (with bounds)
    
    weights = exp_ac / sum_exp
    
    # Compute gradient of LSE w.r.t f
    # grad_lse = conv(weights, f) + corr(weights, f) ?
    # Let's use the FFT trick derived before or just np.convolve if N is small?
    # N=600 is small enough for O(N^2) gradient? 600^2 = 360k. 
    # 1000 iterations * 360k = 360M ops. Might be slow in Python loop.
    # FFT is safer.
    
    # Reusing the gradient logic from the seed but ensuring correctness.
    # grad_lse[i] = sum_k weights[k] * (f[i+k] if i+k<N else 0 + f[i-k] if i-k>=0 else 0)
    
    # Efficient FFT computation:
    nfft = 2 * N
    W = np.fft.rfft(weights, n=nfft)
    F = np.fft.rfft(f, n=nfft)
    
    # Term 1: sum_k w_k f[i+k]
    # This is cross-correlation of f and w.
    # corr[i] = sum_j f[j] w[j-i]
    # Using FFT: ifft(conj(W) * F)
    corr = np.fft.irfft(np.conj(W) * F, n=nfft)
    term1 = corr[:N]
    
    # Term 2: sum_k w_k f[i-k]
    # This is convolution of w and f.
    # conv[i] = sum_k w[k] f[i-k]
    # Using FFT: ifft(W * F)
    conv = np.fft.irfft(W * F, n=nfft)
    term2 = conv[:N]
    
    grad_lse = term1 + term2
    
    s2 = s * s
    s3 = s2 * s
    grad_val = (s * grad_lse - 2.0 * lse) / s3
    
    return val, grad_val
