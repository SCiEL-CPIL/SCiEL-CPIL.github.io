import numpy as np
from scipy.signal import fftconvolve
import pywt

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # Wavelet approach using Haar basis
    # We parameterize the function g(x) by its Haar wavelet coefficients
    # s(x) = sign(g(x))
    
    # Prepare wavelet coefficients
    # Initialize g as a smooth function, e.g., Gaussian
    x = np.linspace(-0.25, 0.25, N)
    g_init = np.exp(-10 * x**2)
    
    # Decompose into wavelet coefficients
    coeffs = pywt.wavedec(g_init, 'haar')
    
    # Flatten coefficients into a single vector for optimization
    # We need to know the lengths to unflatten
    coeff_lens = [len(c) for c in coeffs]
    c_flat = np.concatenate([c.ravel() for c in coeffs])
    
    def reconstruct_g(c_flat):
        # Unflatten coefficients
        coeffs_list = []
        idx = 0
        for length in coeff_lens:
            coeffs_list.append(c_flat[idx:idx+length])
            idx += length
        # idwtl expects the list of coefficients [cA, cD1, cD2, ...]
        g = pywt.idwtl(coeffs_list, 'haar')
        # Ensure length N
        if len(g) < N:
            g = np.pad(g, (0, N - len(g)), mode='edge')
        elif len(g) > N:
            g = g[:N]
        return g

    def compute_score(s):
        # Compute autoconvolution max
        # s is +/- 1
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return np.max(np.abs(r)) * w

    best_c = c_flat.copy()
    best_s = np.sign(reconstruct_g(best_c))
    best_s[best_s == 0] = 1
    best_val = compute_score(best_s)
    
    T = 0.1  # Temperature for tanh
    lr = 0.05
    
    for iteration in range(1000):
        g = reconstruct_g(c_flat)
        
        # Soft sign
        s_soft = np.tanh(g / T)
        
        # Compute autoconvolution of s_soft
        n_fft = 2 * N
        S_soft = np.fft.rfft(s_soft, n_fft)
        r_soft = np.fft.irfft(S_soft * np.conj(S_soft), n_fft)[:N]
        
        # Find max
        abs_r = np.abs(r_soft)
        k_max = np.argmax(abs_r)
        r_max = r_soft[k_max]
        
        # Subgradient of |r[k_max]| w.r.t s_soft
        # r[k] = sum_i s[i] s[i+k]
        # d|r[k]|/ds[i] = sign(r[k]) * (s[i+k] + s[i-k])
        sg_s = np.zeros(N)
        sign_r = np.sign(r_max)
        
        # Vectorized computation of subgradient
        sg_s += np.roll(s_soft, -k_max)
        sg_s += np.roll(s_soft, k_max)
        sg_s *= sign_r
        
        # Gradient w.r.t g
        # d s_soft / d g = (1/T) sech^2(g/T)
        # sech^2(x) = 1 - tanh^2(x)
        d_s_d_g = (1.0 / T) * (1.0 - s_soft**2)
        sg_g = sg_s * d_s_d_g
        
        # Gradient w.r.t c_flat
        # This is the wavelet transform of sg_g
        coeffs_grad = pywt.wavedec(sg_g, 'haar')
        c_grad_flat = np.concatenate([c.ravel() for c in coeffs_grad])
        
        # Update
        c_flat = c_flat - lr * c_grad_flat
        
        # Anneal temperature
        T = max(0.01, T * 0.995)
        lr = 0.05 / (1 + iteration * 0.005)
        
        # Check hard sign score
        s_hard = np.sign(reconstruct_g(c_flat))
        s_hard[s_hard == 0] = 1
        current_val = compute_score(s_hard)
        
        if current_val < best_val:
            best_val = current_val
            best_c = c_flat.copy()
            best_s = s_hard.copy()
            
    return best_s
