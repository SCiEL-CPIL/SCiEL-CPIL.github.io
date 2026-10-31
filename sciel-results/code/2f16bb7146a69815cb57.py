import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: a smooth bump
    # A Gaussian centered at 0 might be good.
    x = np.linspace(-0.25, 0.25, N)
    # f(x) = exp(-x^2 / sigma^2)
    sigma = 0.1
    f_init = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize? The problem doesn't explicitly say to fix the norm.
    # "minimize the normalized maximum of its autoconvolution"
    # Usually "normalized" implies fixing the L1 or L2 norm.
    # If we can scale f, we can make the max autoconvolution arbitrarily small by scaling f down.
    # But f must be nonnegative. If we scale by 0, it's 0.
    # So there must be a constraint.
    # "Choose a nonnegative step function ... to minimize the normalized maximum"
    # Standard interpretation: Fix the L1 norm (area) to 1, or L2 norm to 1.
    # Given the context of "upper bound on C1", this likely refers to a specific inequality where the function is normalized.
    # Let's assume L1 norm = 1.
    # The seed [1.0]*600 has area = 600 * (1/1200) = 0.5.
    # If we normalize the seed to L1=1, it becomes [2.0]*600.
    # Let's check the seed score again.
    # If the evaluator normalizes the input, then the scale doesn't matter.
    # If the evaluator expects the function to have a specific norm, we need to know which one.
    
    # Let's assume the evaluator normalizes the input to have L1 norm 1 or L2 norm 1.
    # Or, the "normalized maximum" is defined as max(f*f) / ||f||_1^2 or similar.
    
    # Let's try to minimize max(f*f) subject to sum(f) * dx = 1.
    # Or simply minimize max(f*f) / (sum(f)*dx)^2.
    
    # Let's define the objective function.
    def objective(f_flat):
        f = f_flat
        # Ensure non-negativity is handled by bounds, but let's clip for safety in intermediate steps?
        # L-BFGS-B handles bounds.
        
        # Compute autoconvolution
        # Discrete correlation
        # c[k] = sum_i f[i] f[i+k]
        # Using FFT is faster for N=600? 
        # FFT length 1200.
        # f_fft = np.fft.rfft(f, n=1200)
        # c_fft = f_fft * np.conj(f_fft)
        # c = np.fft.irfft(c_fft, n=1200)
        # The circular correlation. We need linear correlation.
        # Zero padding to 2N-1 or 2N.
        
        n_fft = 2 * N
        f_pad = np.zeros(n_fft)
        f_pad[:N] = f
        f_fft = np.fft.rfft(f_pad)
        c_fft = f_fft * np.conj(f_fft)
        c = np.fft.irfft(c_fft, n=n_fft)
        
        # The linear correlation is in the first N and last N-1?
        # For auto-correlation of length N, the lags are -N+1 to N-1.
        # The result of irfft will have the positive lags at the beginning and negative at the end (wrapped).
        # Specifically, c[0] is lag 0. c[1] is lag 1. ... c[N-1] is lag N-1.
        # c[N] to c[2N-1] correspond to lags -N+1 to -1?
        # Actually, for real input, the output is symmetric.
        # We just need the max of the absolute values? No, f is non-negative, so f*f is non-negative.
        # So we just take the max of the relevant part.
        
        # The relevant lags are 0 to N-1 (and symmetric).
        # max_val = np.max(c[:N])
        
        # Normalization: Assume L1 norm is fixed to 1.
        # area = np.sum(f) * (0.5 / N)
        # If we don't fix the norm, we can scale f to 0.
        # So we MUST normalize.
        # Let's normalize f to have L1 norm 1.
        area = np.sum(f) * (0.5 / N)
        if area < 1e-10:
            return 1e10
        f_norm = f / area
        # Recalculate autoconvolution of normalized f
        # Or just scale the previous c by 1/area^2
        max_val = np.max(c[:N]) / (area**2)
        
        # Soft max for smoothness?
        # If we use max, it's non-smooth.
        # Let's use a soft max with high beta.
        beta = 10.0
        # max_val is a scalar. We want to minimize it.
        # Wait, c[:N] is a vector. We want to minimize the max element.
        # So objective is max(c[:N]) / area^2.
        
        # To make it smooth:
        # soft_max = (1/beta) * log(sum(exp(beta * c[:N])))
        # But c can be large.
        
        # Let's just return the max for now and use a subgradient method or handle non-smoothness.
        # L-BFGS-B can handle non-smooth to some extent but might struggle.
        # Let's try to minimize the log-sum-exp of the correlation values.
        
        # Shift c to be stable
        c_shifted = c[:N] / (area**2)
        max_c = np.max(c_shifted)
        if max_c > 500:
            return max_c # Avoid overflow
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * (c_shifted - max_c)))) + max_c
        return soft_max

    # Bounds: f_i >= 0
    bounds = [(0, None)] * N
    
    # Options
    options = {'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
    
    # Run optimization
    try:
        res = minimize(objective, f_init, method='L-BFGS-B', bounds=bounds, options=options)
        best_f = res.x
    except Exception:
        best_f = f_init
    
    # Check time
    if time.time() > deadline:
         return list(best_f)
        
    # Polish with more iterations if time permits
    # Maybe try a different initial guess?
    
    return list(best_f)
