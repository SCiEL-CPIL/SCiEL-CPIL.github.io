import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    # Sidelobes are all except the main lobe at center
    # For a step function, the autoconvolution length is 2N-1.
    # The FFT size 2N gives circular convolution. 
    # We need to handle the wrap-around or ensure N is large enough.
    # Standard linear convolution via FFT: size >= 2N-1. 2N is fine.
    # The main lobe is at index N-1 (0-indexed) if we center it?
    # Let's check the index.
    # f is length N. f_fft is length 2N.
    # ac is length 2N.
    # The peak is at lag 0. In the FFT output of f*f, the lag 0 is at index 0? 
    # No, ifft of F^2 gives circular convolution.
    # To get linear convolution, we usually zero-pad.
    # If we pad to 2N, the linear convolution of length 2N-1 fits in 2N.
    # The lag 0 coefficient is at index 0? 
    # Let's trace: 
    # c_k = sum f_i f_{k-i}.
    # If we use fft(f, 2N), the result is periodic.
    # The DC component is at 0.
    # The autoconvolution peak (lag 0) corresponds to the sum of squares, which is the DC component of the spectrum squared? 
    # Actually, the value at lag 0 is max.
    # In the ifft of F^2, the index 0 corresponds to lag 0.
    # Indices 1 to N-1 correspond to positive lags.
    # Indices N to 2N-1 correspond to negative lags (wrap around).
    # So the main lobe is at index 0.
    # Sidelobes are indices 1 to 2N-1, excluding 0.
    # However, due to symmetry, we can just check 1 to N-1 and N to 2N-1.
    # Let's just take all and remove index 0.
    
    sidelobes = np.delete(ac, 0)
    return np.max(np.abs(sidelobes))

def _soft_max_sidelobe(f, lam=100.0):
    """Compute a smooth approximation of the max sidelobe."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    sidelobes = np.delete(ac, 0)
    abs_sidelobes = np.abs(sidelobes)
    
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-10:
        return 0.0
    # Shift to avoid overflow
    shifted = abs_sidelobes - max_val
    return max_val + (1.0/lam) * np.log(np.sum(np.exp(lam * shifted)))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Correct domain: [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N)
    
    # Model: f(x) = exp(-p * x^2) * (1 + q * x^2 + r * x^4)
    # This allows for central dips (if 1 + q*x^2 + r*x^4 < 0 near 0) or peaks.
    # Parameters: p, q, r
    
    def objective(params):
        p, q, r = params
        # p should be positive for decay
        if p < 0:
            return 1e10
        f = np.exp(-p * x**2) * (1 + q * x**2 + r * x**4)
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            return 1e10
        f = f / norm
        return _soft_max_sidelobe(f, lam=50.0)

    best_val = float('inf')
    best_f = None
    
    # Multi-start optimization
    np.random.seed(42)
    
    # Generate a set of initial points
    # p controls width. x is in [-0.25, 0.25], so x^2 is in [0, 0.0625].
    # p * x^2 should range from ~0 to ~10 for significant decay.
    # So p should range from 0 to ~200.
    # q and r control the shape.
    
    num_starts = 100
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        p_init = np.random.uniform(10, 150)
        q_init = np.random.uniform(-20, 20)
        r_init = np.random.uniform(-50, 50)
        
        init = [p_init, q_init, r_init]
        
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                params = res.x
                p, q, r = params
                if p < 0:
                    continue
                f = np.exp(-p * x**2) * (1 + q * x**2 + r * x**4)
                norm = np.linalg.norm(f)
                if norm > 1e-10:
                    best_f = f / norm
        except Exception:
            pass

    # If we found a good candidate, try to polish it with a few discrete random restarts?
    # Or just return the parametric best.
    # The parametric best is likely very close to optimal for this family.
    
    if best_f is None:
        # Fallback: simple Kaiser window on correct domain
        # Kaiser window is defined on [0, 1] typically, but we can map it.
        # Or just use a Gaussian.
        p = 50.0
        f = np.exp(-p * x**2)
        return f / np.linalg.norm(f)
        
    # Final check: ensure the returned vector is valid and normalized
    return best_f
