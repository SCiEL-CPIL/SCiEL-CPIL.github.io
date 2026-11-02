import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def autoconv_normalized(h):
        """Compute normalized autoconvolution for nonnegative h."""
        s = h.sum()
        if s == 0:
            return np.zeros(N), 1e10
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        fft_h = np.fft.fft(h_padded)
        autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        norm_autocorr = autocorr / (s * s)
        return norm_autocorr, norm_autocorr.max()
    
    def objective_smooth(x, beta=50.0):
        """Smooth surrogate: soft-max of normalized autoconvolution.
        Uses h = exp(x) to remove nonnegativity constraint."""
        h = np.exp(x)
        norm_ac, _ = autoconv_normalized(h)
        # Soft-max via log-sum-exp
        max_val = norm_ac.max()
        if max_val <= 0:
            return 1e10
        # Soft max: (1/beta) * log(sum(exp(beta * (norm_ac - max_val)))) + max_val
        shifted = beta * (norm_ac - max_val)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        return max_val + log_sum_exp / beta
    
    def gradient_smooth(x, beta=50.0):
        """Gradient of smooth objective w.r.t. x (where h = exp(x))."""
        h = np.exp(x)
        s = h.sum()
        if s == 0:
            return np.zeros(N)
        
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        fft_h = np.fft.fft(h_padded)
        autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        norm_ac = autocorr / (s * s)
        
        max_val = norm_ac.max()
        if max_val <= 0:
            return np.zeros(N)
        
        shifted = beta * (norm_ac - max_val)
        shifted = np.clip(shifted, -100, 100)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Gradient of soft-max w.r.t. norm_ac
        grad_norm_ac = beta * weights
        
        # Gradient of norm_ac[k] w.r.t. h[i]
        # norm_ac[k] = autocorr[k] / s^2
        # d(autocorr[k])/d(h[i]) = h[i+k] + h[i-k] (with bounds)
        # d(norm_ac[k])/d(h[i]) = (grad_autocorr[i] * s - 2 * autocorr[k] * h[i]) / s^3
        
        grad_h = np.zeros(N)
        for k in range(N):
            wk = grad_norm_ac[k]
            if wk < 1e-15:
                continue
            ac_k = autocorr[k]
            for i in range(N):
                contrib = 0.0
                if i + k < N:
                    contrib += h[i + k]
                if i - k >= 0:
                    contrib += h[i - k]
                grad_h[i] += wk * (contrib * s - 2 * ac_k * h[i]) / (s * s * s)
        
        # Chain rule: d/dx = d/dh * h
        return grad_h * h
    
    def objective_exact(h):
        """Exact objective (hard max) for final evaluation."""
        h = np.maximum(h, 0)
        _, val = autoconv_normalized(h)
        return val
    
    def polish(h_init, maxiter=200):
        """High-precision polish pass."""
        x_init = np.log(np.maximum(h_init, 1e-10))
        try:
            res = minimize(objective_smooth, x_init, jac=gradient_smooth,
                          method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
            h_result = np.exp(res.x)
            return h_result, objective_exact(h_result)
        except:
            return h_init, objective_exact(h_init)
    
    # Initial guesses
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = objective_exact(h0)
    
    # Try different initializations with smooth objective
    inits = [
        np.ones(N),
        np.sin(np.pi * np.arange(N) / N) ** 2,
        np.exp(-((np.arange(N) - N/2) / (N/4)) ** 2),
        np.exp(-((np.arange(N) - N/2) / (N/6)) ** 2),
        np.ones(N) * (1 + 0.1 * np.sin(2 * np.pi * np.arange(N) / N)),
        np.ones(N) * (1 + 0.1 * np.cos(2 * np.pi * np.arange(N) / N)),
    ]
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        h_init = np.maximum(h_init, 1e-10)
        if h_init.sum() == 0:
            continue
        
        x_init = np.log(h_init)
        try:
            res = minimize(objective_smooth, x_init, jac=gradient_smooth,
                          method='L-BFGS-B',
                          options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            h_result = np.exp(res.x)
            val = objective_exact(h_result)
            if val < best_val:
                best_h = h_result.copy()
                best_val = val
        except:
            pass
    
    # Polish pass on best solution
    if time.time() < deadline - 10:
        polished_h, polished_val = polish(best_h, maxiter=150)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    return list(np.maximum(best_h, 0))
