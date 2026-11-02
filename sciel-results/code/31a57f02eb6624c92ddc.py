"""Spectral phase optimization for minimizing max autoconvolution."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_from_spectrum(X, L):
    """Compute linear autoconvolution from zero-padded spectrum X."""
    C = np.fft.irfft(X * X, n=L)
    return C

def _objective_and_grad(params, L, beta=50.0):
    """
    params: array of phases for k=1 to L/2-1
    Returns: (objective_value, gradient)
    """
    X = np.zeros(L // 2 + 1, dtype=complex)
    X[0] = 1.0
    
    for i, phi in enumerate(params):
        X[i + 1] = np.exp(1j * phi)
    
    C = _autoconv_from_spectrum(X, L)
    abs_C = np.abs(C)
    max_C = np.max(abs_C)
    
    if max_C < 1e-10:
        return 1e-10, np.zeros_like(params)
    
    shifted = beta * (abs_C - max_C)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_C + (1.0 / beta) * np.log(sum_exp)
    
    W = weights * np.sign(C)
    
    W_full = np.zeros(L, dtype=complex)
    W_full[:L//2+1] = W
    W_full[L//2+1:] = W[L//2:0:-1]
    
    Z_full = np.fft.ifft(W_full)
    
    grad = np.zeros(len(params))
    for i, k in enumerate(range(1, L//2)):
        X_k = X[k]
        Z_k = Z_full[k]
        grad[i] = -4.0 * np.imag(Z_k * X_k * X_k)
        
    return soft_max, grad

def _spectral_optimize(L, seed=42, max_iters=100, beta=50.0):
    n_phases = L // 2 - 1
    if n_phases <= 0:
        return np.ones(N) / N
        
    rng = np.random.RandomState(seed)
    phi0 = rng.uniform(0, 2 * np.pi, size=n_phases)
    
    res = minimize(
        _objective_and_grad,
        phi0,
        args=(L, beta),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iters, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    X = np.zeros(L // 2 + 1, dtype=complex)
    X[0] = 1.0
    for i, phi in enumerate(res.x):
        X[i + 1] = np.exp(1j * phi)
    
    x_full = np.fft.irfft(X, n=L)
    x = x_full[:N]
    
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    else:
        x = np.ones(N) / N
        
    return x

def construct():
    L = 1
    while L < 2 * N - 1:
        L *= 2
        
    best_x = None
    best_val = 1e10
    
    # Try different betas and seeds
    for beta in [30, 50, 100]:
        if _check_deadline():
            break
            
        for seed in [42, 123, 999, 31415]:
            if _check_deadline():
                break
            
            x = _spectral_optimize(L, seed=seed, max_iters=150, beta=beta)
            
            # Calculate raw score
            X = np.fft.rfft(x, n=L)
            C = np.fft.irfft(X * X, n=L)[:2*N-1]
            s = np.sum(x)
            if abs(s) < 1e-10:
                val = 1e10
            else:
                val = np.max(np.abs(C)) / (s * s)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
                
    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
