import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    # 150s limit, keep 20s margin for safety
    deadline = start_time + 130.0
    
    # 1. Continuous Relaxation Strategy
    # We optimize a continuous vector x in [-1, 1] to minimize the energy of the autocorrelation
    # R[k] = sum x[i] x[i+k]
    # Cost function: Sum of squares of non-zero lags. This is smooth and differentiable.
    
    def cost(x):
        # x is a flat array of length N
        # Compute autocorrelation via FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        power = np.abs(F) ** 2
        R_full = np.fft.irfft(power, n=n_fft)
        R = R_full[:N]
        # R[0] is the DC term. We want to minimize energy in non-zero lags.
        # Note: R[0] is constant if x is binary, but varies if x is continuous.
        # However, for the metric C3, we care about max |R[k]| / R[0].
        # Minimizing sum of squares is a good proxy for reducing peaks.
        return np.sum(R[1:] ** 2)

    # Initialize
    np.random.seed(42)
    x0 = np.random.uniform(-1, 1, N)
    
    # Optimize
    res = None
    try:
        if time.time() < deadline - 30.0:
            res = minimize(cost, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
    except Exception:
        pass
    
    if res is not None and (res.success or time.time() < deadline - 20.0):
        x_opt = res.x
    else:
        x_opt = x0

    # Discretize
    h = np.sign(x_opt)
    # Handle zeros (unlikely but possible)
    h[h == 0] = 1.0
    
    # 2. Discrete Refinement (Bit Flipping)
    # This fixes the logic from the previous failed program but ensures correct shapes.
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def compute_R(h):
        F = np.fft.rfft(h, n=n_fft)
        power = np.abs(F) ** 2
        R_full = np.fft.irfft(power, n=n_fft)
        R = R_full[:N].copy()
        R[0] = float(N) # Ensure exact DC for binary
        return R

    R = compute_R(h)
    
    def get_metric(R):
        if len(R) <= 1:
            return 0.0
        return np.max(np.abs(R[1:])) / R[0]

    best_h = h.copy()
    best_val = get_metric(R)
    
    k = np.arange(1, N) # Length N-1
    
    # Greedy Local Search
    improved = True
    passes = 0
    max_passes = 20 # Limit passes to save time for potential restarts
    
    while improved and passes < max_passes:
        if time.time() > deadline - 10.0:
            break
        improved = False
        passes += 1
        
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            
            # Calculate delta for flipping h[i]
            # R_new[k] = R[k] - 2*h[i]*(h[i+k] + h[i-k]) for k=1..N-1
            # We need to handle boundaries.
            
            # term1: h[i+k] for k=1..N-1
            # Indices: i+1, i+2, ..., i+N-1
            # Valid if index < N
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            if np.any(valid1):
                term1[valid1] = h[idx1[valid1]]
                
            # term2: h[i-k] for k=1..N-1
            # Indices: i-1, i-2, ..., i-(N-1)
            # Valid if index >= 0
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            if np.any(valid2):
                term2[valid2] = h[idx2[valid2]]
            
            delta = -2.0 * h[i] * (term1 + term2) # Shape (N-1,)
            
            # Update R[1:]
            new_R = R.copy()
            new_R[1:] = R[1:] + delta
            
            new_val = get_metric(new_R)
            
            if new_val < best_val:
                h[i] *= -1.0
                R = new_R
                best_val = new_val
                best_h = h.copy()
                improved = True

    # 3. Optional: Random Restarts if time permits
    if time.time() < deadline - 40.0:
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb best_h
            h_curr = best_h.copy()
            flip_mask = np.random.rand(N) < 0.05 # Flip 5%
            h_curr[flip_mask] *= -1.0
            
            R_curr = compute_R(h_curr)
            curr_val = get_metric(R_curr)
            
            # Quick greedy pass
            improved = True
            passes = 0
            while improved and passes < 5:
                if time.time() > deadline - 10.0:
                    break
                improved = False
                passes += 1
                for i in range(N):
                    if time.time() > deadline - 10.0:
                        break
                    idx1 = i + k
                    valid1 = idx1 < N
                    term1 = np.zeros(N-1)
                    if np.any(valid1):
                        term1[valid1] = h_curr[idx1[valid1]]
                    
                    idx2 = i - k
                    valid2 = idx2 >= 0
                    term2 = np.zeros(N-1)
                    if np.any(valid2):
                        term2[valid2] = h_curr[idx2[valid2]]
                    
                    delta = -2.0 * h_curr[i] * (term1 + term2)
                    new_R = R_curr.copy()
                    new_R[1:] = R_curr[1:] + delta
                    new_val = get_metric(new_R)
                    
                    if new_val < curr_val:
                        h_curr[i] *= -1.0
                        R_curr = new_R
                        curr_val = new_val
                        improved = True
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()

    return best_h
