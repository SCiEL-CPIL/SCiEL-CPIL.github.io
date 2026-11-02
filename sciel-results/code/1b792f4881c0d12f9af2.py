"""Erdos minimum overlap - Continuous Gradient Descent for Min Overlap."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET = N / 2.0

def _calc_autocorr(h):
    h_fft = np.fft.fft(h)
    c = np.fft.ifft(h_fft * np.conj(h_fft)).real
    return c

def _objective(h):
    c = _calc_autocorr(h)
    overlaps = TARGET - c[1:]
    # Sum of exponentials of overlaps
    # Scale to prevent overflow
    beta = 10.0
    return np.sum(np.exp(beta * overlaps))

def _gradient(h):
    c = _calc_autocorr(h)
    overlaps = TARGET - c[1:]
    beta = 10.0
    weights = beta * np.exp(beta * overlaps)
    
    # grad_h[i] = sum_{k=1}^{N-1} weights[k] * (- (h[i-k] + h[i+k]))
    # Note: c_k = sum_j h_j h_{j+k}
    # d c_k / d h_i = h_{i-k} + h_{i+k}
    # d obj / d h_i = sum_k weights[k] * (- d c_k / d h_i)
    
    grad = np.zeros(N)
    for k in range(1, N):
        if weights[k] < 1e-6:
            continue
        # h[i-k] is roll(h, k)
        # h[i+k] is roll(h, -k)
        grad -= weights[k] * (np.roll(h, k) + np.roll(h, -k))
        
    return grad

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Binary alternating
    h0 = np.zeros(N)
    h0[::2] = 1.0
    
    best_h = h0.copy()
    best_val = np.inf
    
    # Run Gradient Descent
    # Use L-BFGS-B with bounds [0, 1]
    bounds = [(0, 1)] * N
    
    # We might need multiple restarts if it gets stuck
    for i in range(3):
        if time.time() > deadline:
            break
            
        h_init = best_h.copy() if i > 0 else h0.copy()
        
        # Add small noise for restarts
        if i > 0:
            noise = rng.random(N) * 0.1
            h_init = np.clip(h_init + noise - 0.05, 0, 1)
            # Normalize sum to N/2? 
            # The problem requires sum = N/2.
            # Gradient descent will drift. We must project or constrain.
            # L-BFGS-B doesn't handle equality constraints well.
            # We can use a penalty method or normalize after each step.
            
        # Custom callback to normalize and check time
        def callback(hk):
            # Normalize to sum N/2
            s = np.sum(hk)
            if s > 0:
                hk = hk * (TARGET / s)
            # Clip to [0, 1]
            np.clip(hk, 0, 1, out=hk)
            # Re-normalize after clip
            s2 = np.sum(hk)
            if s2 > 0:
                hk = hk * (TARGET / s2)
            
            if time.time() > deadline:
                raise RuntimeError("Timeout")
            
            return False

        try:
            res = minimize(_objective, h_init, jac=_gradient, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-6, 'callback': callback})
            if res.success or res.status == 0:
                h_final = res.x
                # Ensure constraints
                s = np.sum(h_final)
                h_final = h_final * (TARGET / s)
                np.clip(h_final, 0, 1, out=h_final)
                s2 = np.sum(h_final)
                if s2 > 0:
                    h_final = h_final * (TARGET / s2)
                
                val = _objective(h_final)
                if val < best_val:
                    best_val = val
                    best_h = h_final.copy()
        except Exception as e:
            pass

    return best_h
