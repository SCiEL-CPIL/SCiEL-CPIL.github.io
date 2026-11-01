import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _objective_and_gradient(x):
    """
    Objective: Sum of squares of sidelobes.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    # Sidelobes are ac[1:N]
    sidelobes = ac[1:N]
    obj = np.sum(sidelobes**2)
    
    # Gradient calculation
    # d/dx_i (sum_k C[k]^2) = 2 * sum_k C[k] * dC[k]/dx_i
    # dC[k]/dx_i = h[i+k] + h[i-k] (if valid)
    
    grad = np.zeros(N)
    # We can compute this efficiently.
    # For each k, the contribution to grad is 2 * C[k] * (h[i+k] + h[i-k])
    # This is a convolution-like operation.
    # Let's just loop over k, it's 399 iterations.
    for k in range(1, N):
        c_k = sidelobes[k-1]
        if c_k == 0:
            continue
        # Contribution to indices i where i+k < N: h[i+k]
        # Contribution to indices i where i-k >= 0: h[i-k]
        
        # Vectorized update
        # Indices i such that i+k < N => i < N-k
        # grad[0:N-k] += 2 * c_k * h[k:N]
        # Indices i such that i-k >= 0 => i >= k
        # grad[k:N] += 2 * c_k * h[0:N-k]
        
        grad[0:N-k] += 2.0 * c_k * h[k:N]
        grad[k:N] += 2.0 * c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            # Revert
            h[i] = old_h_i
            # ac remains unchanged (old ac)
            
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation
    # Try a few random initializations
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline:
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N) * 0.9
        
        # Use L-BFGS-B
        try:
            res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-5, 'gtol': 1e-5})
        except Exception:
            continue
        
        if res.success or res.fun < best_val: # Note: res.fun is sum of squares, not max sidelobe
            # Round to binary
            h_bin = np.sign(res.x)
            # Handle zeros
            h_bin[h_bin == 0] = np.random.choice([-1.0, 1.0], size=np.sum(h_bin == 0))
            
            # Evaluate binary solution
            ac = _compute_autocorr(h_bin)
            val = _max_sidelobe(ac)
            
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
        # Refine the best binary solution found so far using local search
        # We can do this after each continuous start or just at the end.
        # Let's do a quick refinement here.
        if best_h is not None:
            if time.time() < deadline - 10.0: # Reserve 10s for final search
                h_ref, val_ref = _local_search(best_h.copy(), max_iterations=2000, time_limit=2.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 2: Final Discrete Search
    # If we have time left, run a more extensive discrete search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use the best_h as starting point for a longer local search
        # Or run multiple random restarts if we have lots of time
        h_final, val_final = _local_search(best_h.copy(), max_iterations=100000, time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
