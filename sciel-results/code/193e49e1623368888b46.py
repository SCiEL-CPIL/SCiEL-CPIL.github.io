import numpy as np
from scipy.signal import windows
from scipy.optimize import minimize
import time

N = 400

def _autoconv_bound(f):
    """
    Compute the upper bound on C3 for a signed step function.
    The bound is the maximum absolute value of the autoconvolution sidelobes.
    """
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def _soft_max_sidelobe(f, lam=20.0):
    """Compute a smooth approximation of the max sidelobe."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    abs_sidelobes = np.abs(sidelobes)
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_sidelobes - max_val
    return max_val + (1.0/lam) * np.log(np.sum(np.exp(lam * shifted)))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Strategy: Use a sum of low-frequency sines (band-limited signal) to create smooth, oscillating profiles.
    # This leverages the spectral properties of smooth functions to minimize the autoconvolution tail.
    # We model f(x) = sum_{k=1}^{K} A_k * sin(2*pi*k*x + phase_k)
    # We optimize the amplitudes and phases.
    
    K = 4  # Number of sine components
    
    def objective(params):
        # params: [A_1, phi_1, A_2, phi_2, ..., A_K, phi_K]
        f = np.zeros(N)
        for i in range(K):
            A = params[2*i]
            phi = params[2*i+1]
            f += A * np.sin(2*np.pi*(i+1)*x + phi)
        
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            return 1e10
        f = f / norm
        return _soft_max_sidelobe(f, lam=25.0)
    
    best_val = float('inf')
    best_f = None
    
    # Multiple random restarts with different initial frequencies/amplitudes
    rng = np.random.default_rng(42)
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        # Amplitudes in [0.5, 2.0], phases in [0, 2*pi]
        init_params = []
        for k in range(K):
            A = rng.uniform(0.5, 2.0)
            phi = rng.uniform(0, 2*np.pi)
            init_params.extend([A, phi])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                params = res.x
                f = np.zeros(N)
                for k in range(K):
                    A = params[2*k]
                    phi = params[2*k+1]
                    f += A * np.sin(2*np.pi*(k+1)*x + phi)
                norm = np.linalg.norm(f)
                if norm > 1e-10:
                    best_f = f / norm
        except Exception:
            pass
    
    # If sine model didn't work well, try a hybrid with Kaiser window or direct optimization
    if best_f is None or best_val > 0.1:
        # Fallback: Direct optimization of a few parameters in a more flexible basis
        # Use a cosine series: f(x) = sum A_k cos(2*pi*k*x)
        def objective_cos(params):
            f = np.zeros(N)
            for k in range(K):
                A = params[k]
                f += A * np.cos(2*np.pi*(k+1)*x)
            norm = np.linalg.norm(f)
            if norm < 1e-10:
                return 1e10
            f = f / norm
            return _soft_max_sidelobe(f, lam=25.0)
        
        for i in range(5):
            if time.time() > deadline - 10:
                break
            init_params = rng.uniform(-1, 1, K)
            try:
                res = minimize(objective_cos, init_params, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    params = res.x
                    f = np.zeros(N)
                    for k in range(K):
                        A = params[k]
                        f += A * np.cos(2*np.pi*(k+1)*x)
                    norm = np.linalg.norm(f)
                    if norm > 1e-10:
                        best_f = f / norm
            except Exception:
                pass
    
    # If still no good result, fallback to Kaiser
    if best_f is None:
        best_kaiser_val = float('inf')
        best_kaiser_f = None
        for beta in np.arange(4, 24, 2):
            if time.time() > deadline - 5:
                break
            f = windows.kaiser(N, beta)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _autoconv_bound(f)
            if val < best_kaiser_val:
                best_kaiser_val = val
                best_kaiser_f = f.copy()
        
        if best_kaiser_f is not None:
            return best_kaiser_f
        else:
            f = windows.kaiser(N, 12)
            return f / np.linalg.norm(f)
    
    # Final polish: Local refinement around the best found solution
    if time.time() < deadline - 15 and best_f is not None:
        # Re-optimize with the best found parameters as initial guess
        # We need to reconstruct the parameters from the best_f, which is complex.
        # Instead, we just return the best_f found so far.
        pass
    
    return best_f
