import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: constant 0.5
    # We parameterize h using a small number of Fourier coefficients to ensure smoothness
    # which might help if the metric rewards smoothness or if my analysis of the min-correlation is flawed for specific discrete k.
    
    def calculate_overlap(h):
        # h is array of length N
        # Calculate sum_i h_i * (1 - h_{i+k}) for all k
        # Using FFT for speed
        # Correlation of h and 1-h
        # Let g = 1 - h
        # overlap_k = sum_i h_i * g_{i+k}
        # This is cross-correlation.
        # We want max_k overlap_k.
        # To get all shifts, we can use np.roll.
        # For N=100, direct computation is O(N^2) = 10000, which is fast.
        overlaps = []
        for k in range(N):
            h_shifted = np.roll(h, -k)
            # overlap is sum(h * (1 - h_shifted))
            ov = np.sum(h * (1 - h_shifted))
            overlaps.append(ov)
        return np.max(overlaps)
    
    def objective(params):
        # params: [a0, a1, b1, ..., aM, bM]
        # h_i = a0 + sum (a_j cos(2 pi j i / N) + b_j sin(2 pi j i / N))
        # We need h_i in [0,1] and sum h_i = N/2.
        # a0 is the mean. sum h_i = N * a0 + ... = N/2 => a0 = 0.5 if other terms sum to 0.
        # If we use cosine/sine, their sum over 0..N-1 is 0.
        # So we fix a0 = 0.5.
        # We optimize the amplitudes of the harmonics.
        # To ensure h_i in [0,1], we will clip and rescale.
        
        M = 5 # Number of harmonics
        # params length is 2*M
        coeffs = params
        i = np.arange(N)
        h = np.full(N, 0.5)
        for j in range(1, M+1):
            a = coeffs[2*(j-1)]
            b = coeffs[2*(j-1)+1]
            h += a * np.cos(2 * np.pi * j * i / N) + b * np.sin(2 * np.pi * j * i / N)
        
        # Clip to [0,1]
        h = np.clip(h, 0, 1)
        
        # Normalize sum to N/2
        s = np.sum(h)
        if s > 0:
            h = h * (N / 2.0) / s
        else:
            h = np.full(N, 0.5)
            
        # Re-clip after scaling (scaling might push out of bounds if s was small, but s is close to N/2)
        # Actually, if we clip then scale, the sum is exactly N/2, but some values might exceed 1.
        # We need a projection that respects [0,1] and sum=N/2.
        # Simple iterative projection:
        for _ in range(10):
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
            else:
                h = np.full(N, 0.5)
                break
        
        return calculate_overlap(h)

    # Initial parameters: all zeros (constant 0.5)
    M = 5
    x0 = np.zeros(2*M)
    
    best_score = calculate_overlap(np.full(N, 0.5))
    best_h = np.full(N, 0.5)
    
    # Try a few random starts and Nelder-Mead
    # Budget: ~10 seconds for optimization
    time_limit_for_opt = 10.0
    
    for _ in range(20):
        if time.time() > start_time + time_limit_for_opt:
            break
        
        # Random small perturbation
        x0 = np.random.normal(0, 0.1, 2*M)
        
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success or res.fun < best_score:
            # Reconstruct h from best params
            # Note: objective returns the score, but we need the h that produced it.
            # We can re-run the reconstruction logic inside or return it.
            # Let's refactor to return h as well or store it.
            pass # Simplified: just track best score. We need the best h.
            
    # Refinement: Local search on the discrete h
    # Start from constant 0.5 or best found
    h = best_h.copy()
    
    # Simple local search: try changing one element
    # To make it faster, we can use a gradient approximation or just random swaps.
    # Given N=100, we can do a few iterations.
    
    def get_h_from_params(params):
        M = 5
        coeffs = params
        i = np.arange(N)
        h = np.full(N, 0.5)
        for j in range(1, M+1):
            a = coeffs[2*(j-1)]
            b = coeffs[2*(j-1)+1]
            h += a * np.cos(2 * np.pi * j * i / N) + b * np.sin(2 * np.pi * j * i / N)
        
        for _ in range(10):
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
            else:
                h = np.full(N, 0.5)
                break
        return h

    # Run optimization properly
    best_params = np.zeros(2*M)
    best_val = calculate_overlap(best_h)
    
    for attempt in range(30):
        if time.time() > start_time + 10.0:
            break
        x0 = np.random.normal(0, 0.2, 2*M)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    h = get_h_from_params(best_params)
    
    # Final polish: Local discrete search
    # Try to improve by adjusting individual h_i
    # Calculate overlaps once, then update efficiently?
    # For N=100, full recalc is 10000 ops. Very fast.
    
    for step in range(1000):
        if time.time() > deadline:
            break
        i = np.random.randint(N)
        old_val = h[i]
        # Try a small change
        delta = np.random.uniform(-0.1, 0.1)
        new_val = np.clip(old_val + delta, 0, 1)
        
        h_new = h.copy()
        h_new[i] = new_val
        
        # Normalize sum
        s = np.sum(h_new)
        if s > 0:
            h_new = h_new * (N / 2.0) / s
        else:
            continue
            
        # Re-clip
        h_new = np.clip(h_new, 0, 1)
        
        score_new = calculate_overlap(h_new)
        if score_new < best_val:
            best_val = score_new
            h = h_new
    
    return h

# EVOLVE-BLOCK-END
