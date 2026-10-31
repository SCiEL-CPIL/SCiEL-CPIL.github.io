import numpy as np
import scipy.optimize
import time

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _objective_l2(f):
    """
    Objective function for optimization: Sum of squared sidelobes.
    This is differentiable and smooth.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    sidelobes = R[1:N]
    return np.sum(sidelobes**2)

def _objective_grad(f):
    """
    Gradient of the L2 objective.
    Derivation:
    R[k] = sum_t f[t] f[t+k]
    d(R[k]^2)/df[i] = 2 R[k] * d(R[k])/df[i]
    d(R[k])/df[i] is non-zero if i is in [0, N-1] and i+k is in [0, N-1].
    Specifically, R[k] = sum_{t=0}^{N-1-k} f[t] f[t+k]
    d(R[k])/df[i] = f[i+k] (if i < N-k) + f[i-k] (if i >= k)
    
    This is computationally expensive to compute analytically for every step.
    We can use numerical gradients or rely on L-BFGS-B's internal approximations if we don't provide grad.
    However, for N=400, numerical gradients (finite differences) might be slow (400 function evals per gradient step).
    L-BFGS-B with analytical gradient is preferred.
    
    Let's try to compute the gradient efficiently using FFTs.
    Actually, the gradient of the energy of the sidelobes is related to the convolution of f with the sidelobe error.
    
    Alternatively, we can just use `method='Nelder-Mead'` which doesn't need gradients, but might be slow in 400D.
    Or `method='Powell'`.
    
    Let's stick to L-BFGS-B and provide a numerical gradient if analytical is too complex, or just let scipy approximate it.
    Actually, scipy.optimize.minimize with L-BFGS-B will approximate the gradient using finite differences if not provided.
    400 dimensions * 2 (forward/backward) = 800 function evaluations per gradient step.
    Each function eval is an FFT of size 800. Very fast (~microseconds).
    So 800 * 1us = 0.8ms per gradient step.
    We can afford hundreds of iterations.
    """
    pass # Not used directly, relying on scipy's approximation or implementing analytical if needed.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_f = None
    best_score = float('inf')

    # Try a few initial window shapes
    # 1. Kaiser window
    for beta in [6, 8, 10, 12, 14]:
        if time.time() > deadline - 10:
            break
        f_init = scipy.signal.windows.kaiser(N, beta=beta)
        # Normalize to unit energy
        f_init = f_init / np.linalg.norm(f_init)
        
        # Optimize
        try:
            res = scipy.optimize.minimize(
                _objective_l2, 
                f_init, 
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            f_opt = res.x
            # Re-normalize to unit energy to ensure consistent scaling
            f_opt = f_opt / np.linalg.norm(f_opt)
            
            score = _compute_bound(f_opt)
            
            if score < best_score:
                best_score = score
                best_f = f_opt.copy()
                
        except Exception as e:
            continue

    # Fallback if optimization fails or returns bad result
    if best_f is None or best_score > 1.0:
        # Use a simple Gaussian window
        x = np.linspace(-1, 1, N)
        f_init = np.exp(-2 * x**2)
        f_init = f_init / np.linalg.norm(f_init)
        best_f = f_init
        best_score = _compute_bound(best_f)

    return best_f
