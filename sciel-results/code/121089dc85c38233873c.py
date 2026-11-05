import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Objective: minimize max of convolution
    # Using log-sum-exp for smoothness
    beta = 10.0 # Temperature for max approximation
    
    def objective(params):
        # params are Fourier coefficients
        # h[0] is DC
        # We use real and imaginary parts of complex coefficients?
        # Let's use real cosine and sine terms.
        # h[i] = a0 + sum_k (ak cos(2pi k i / N) + bk sin(2pi k i / N))
        
        a0 = params[0]
        rest = params[1:]
        
        # Number of modes
        # Let's try 10 modes
        n_modes = 10
        if len(rest) < 2 * n_modes:
             # Pad if necessary, though we fix size
             pass
            
        h = np.zeros(N)
        h[:] = a0
        
        # Reshape rest into (n_modes, 2)
        coeffs = rest[:2*n_modes].reshape(n_modes, 2)
        
        i = np.arange(N)
        for k in range(n_modes):
            ak, bk = coeffs[k]
            h += ak * np.cos(2 * np.pi * k * i / N)
            h += bk * np.sin(2 * np.pi * k * i / N)
            
        # Enforce sum(h) = 1
        # The DC term a0 contributes N * a0 to the sum.
        # The AC terms sum to 0.
        # So sum(h) = N * a0.
        # We want sum(h) = 1 => a0 = 1/N.
        # But we are optimizing a0.
        # To handle the constraint, we can just normalize h at the end?
        # No, the objective depends on the scale.
        # V is scale invariant.
        # So we can fix the sum to 1 during optimization.
        # If we fix a0 = 1/N, we only optimize AC coefficients.
        
        # Let's restart with fixed DC.
        return 0.0 # Placeholder
        
    # Better approach:
    # Fix h such that sum(h)=1.
    # h = 1/N + sum AC terms.
    # Optimize AC coefficients.
    
    n_modes = 10
    x0 = np.zeros(2 * n_modes)
    # Small random init
    x0 = np.random.normal(0, 0.1, size=2 * n_modes)
    
    def obj_and_grad(x):
        # x are AC coefficients
        h = np.ones(N) / N
        i = np.arange(N)
        coeffs = x.reshape(n_modes, 2)
        
        # Compute h
        for k in range(n_modes):
            ak, bk = coeffs[k]
            h += ak * np.cos(2 * np.pi * k * i / N)
            h += bk * np.sin(2 * np.pi * k * i / N)
            
        # Compute convolution
        # Use FFT for speed
        H = np.fft.fft(h, n=2*N)
        Y = H * H
        y = np.fft.ifft(Y).real[:2*N-1]
        
        # Max of y
        # Log-sum-exp
        max_y = np.max(y)
        # If max_y is very large, exp(beta * y) overflows
        # Shift by max_y
        shifted = beta * (y - max_y)
        sum_exp = np.sum(np.exp(shifted))
        val = max_y + np.log(sum_exp) / beta
        
        # Gradient
        # This is complex to derive analytically for FFT.
        # Use finite differences or just minimize without gradient (Nelder-Mead)?
        # L-BFGS-B requires gradient.
        # Let's try to compute gradient via FFT properties?
        # Or just use a few iterations of coordinate descent / random search?
        
        return val, None # Gradient not provided
        
    # Since gradient is hard, use Nelder-Mead or Powell?
    # Nelder-Mead is slow for 20 dims.
    # Powell is okay.
    
    best_val = np.inf
    best_x = x0
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        x0 = np.random.normal(0, 0.5, size=2 * n_modes)
        
        res = minimize(obj_and_grad, x0, method='Powell', options={'maxiter': 100, 'xtol': 1e-4})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        if time.time() > deadline:
            break

    # Reconstruct h from best_x
    h = np.ones(N) / N
    i = np.arange(N)
    coeffs = best_x.reshape(n_modes, 2)
    for k in range(n_modes):
        ak, bk = coeffs[k]
        h += ak * np.cos(2 * np.pi * k * i / N)
        h += bk * np.sin(2 * np.pi * k * i / N)
        
    # Final polish: ensure sum is exactly 1 (it should be, as AC sums to 0)
    # But numerical errors might creep in.
    h /= np.sum(h)
    
    return h
