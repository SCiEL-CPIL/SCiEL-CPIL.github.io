import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_score_fast(h):
    """Compute the maximum overlap of h with shifted 1-h using FFT."""
    # h is in [0,1], sum = N/2
    # overlap(s) = sum_i h[i] * (1 - h[(i+s) % N])
    #            = sum_i h[i] - sum_i h[i] * h[(i+s) % N]
    #            = N/2 - autocorr(s)
    # where autocorr(s) = sum_i h[i] * h[(i+s) % N]
    
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h)
    acf = np.real(np.fft.ifft(H * np.conj(H)))
    # acf[s] = sum_i h[i] * h[(i-s) % N] ... need to be careful with indexing
    # Actually np.fft.ifft of |H|^2 gives circular autocorrelation
    # acf[k] = sum_j h[j] * h[(j+k) % N]
    
    max_ov = N/2 - np.min(acf)
    return max_ov

def project_to_feasible(h):
    """Project h to satisfy sum = N/2 and 0 <= h[i] <= 1."""
    h = h.copy()
    # First clip
    h = np.clip(h, 0, 1)
    # Adjust sum
    target = N / 2
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute uniformly
        h += diff / N
        h = np.clip(h, 0, 1)
        # If still off, do a second pass
        diff2 = target - np.sum(h)
        if abs(diff2) > 1e-12:
            # Only adjust non-bound elements
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.sum(mask) > 0:
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0, 1)
    # Final exact adjustment
    diff3 = target - np.sum(h)
    if abs(diff3) > 1e-14:
        # Spread over all elements
        h += diff3 / N
        h = np.clip(h, 0, 1)
    return h

def construct():
    deadline = time.time() + 130  # Leave 20s margin
    
    # Start with uniform
    h = np.full(N, 0.5)
    best_score = compute_score_fast(h)
    best_h = h.copy()
    
    # Try some structured initializations
    inits = []
    
    # Alternating
    alt = np.tile([0.0, 1.0], N // 2)
    inits.append(alt)
    
    # Block patterns
    for k in [1, 2, 3, 5, 7, 10]:
        block = np.zeros(N)
        # Create periodic blocks
        period = N // k
        for i in range(k):
            block[i*period:(i+1)*period] = 1.0
        # This has sum = k * period = N, need N/2
        # So make half of each block 0
        for i in range(k):
            start = i * period
            block[start:start+period//2] = 1.0
            block[start+period//2:start+period] = 0.0
        inits.append(block)
    
    # Random initializations
    for _ in range(20):
        r = np.random.uniform(0, 1, N)
        inits.append(r)
    
    # Evaluate all initializations
    for init in inits:
        h_test = project_to_feasible(init)
        score = compute_score_fast(h_test)
        if score < best_score:
            best_score = score
            best_h = h_test.copy()
    
    # Local search with gradient-based optimization
    # Objective: maximize min autocorrelation (equivalent to minimizing max overlap)
    # Use scipy minimize on -min(acf)
    
    def objective(x):
        h_local = project_to_feasible(x)
        H = np.fft.fft(h_local)
        acf = np.real(np.fft.ifft(H * np.conj(H)))
        return -np.min(acf)
    
    # Multiple restarts from best and random points
    starts = [best_h.copy()]
    for _ in range(10):
        starts.append(np.random.uniform(0, 1, N))
    
    for start in starts:
        if time.time() > deadline:
            break
        
        x0 = project_to_feasible(start)
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            h_opt = project_to_feasible(result.x)
            score = compute_score_fast(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional local search: coordinate descent
    if time.time() < deadline:
        h_cd = best_h.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try small perturbations
                for delta in [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]:
                    h_try = h_cd.copy()
                    h_try[i] += delta
                    h_try = project_to_feasible(h_try)
                    score_try = compute_score_fast(h_try)
                    if score_try < best_score - 1e-10:
                        best_score = score_try
                        best_h = h_try.copy()
                        h_cd = h_try.copy()
                        improved = True
                        break
    
    return best_h
