import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 120.0  # Safety margin from 150s limit

def compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h (linear, no wrap)."""
    N = len(h)
    one_minus_h = 1.0 - h
    max_val = 0.0
    for k in range(N):
        n = N - k
        s = np.dot(h[:n], one_minus_h[k:k+n])
        if s > max_val:
            max_val = s
    return max_val

def objective(x, penalty_lambda=1e6):
    """Objective: max_overlap + penalty for sum constraint."""
    h = np.clip(x, 0, 1)
    overlap = compute_max_overlap(h)
    sum_dev = np.sum(h) - N / 2.0
    return overlap + penalty_lambda * sum_dev * sum_dev

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = compute_max_overlap(best_h)
    
    # Helper to check deadline
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    
    # Start 1: uniform 0.5
    starts = [np.full(N, 0.5)]
    
    # Start 2: random
    for _ in range(10):
        if time_left() < 20:
            break
        h_rand = rng.random(N)
        # Scale to sum N/2
        h_rand = h_rand / np.sum(h_rand) * (N / 2.0)
        h_rand = np.clip(h_rand, 0, 1)
        # Adjust sum
        diff = N / 2.0 - np.sum(h_rand)
        if diff > 0:
            # Add to elements below 1
            mask = h_rand < 1.0
            if np.sum(mask) > 0:
                h_rand[mask] += diff / np.sum(mask)
                h_rand = np.clip(h_rand, 0, 1)
        else:
            # Subtract from elements above 0
            mask = h_rand > 0.0
            if np.sum(mask) > 0:
                h_rand[mask] += diff / np.sum(mask)
                h_rand = np.clip(h_rand, 0, 1)
        starts.append(h_rand.copy())
    
    # Start 3: Fourier modes
    for f in range(1, 10):
        if time_left() < 20:
            break
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                h_fourier = 0.5 + amp * np.cos(2 * np.pi * f * np.arange(N) / N + phase)
                h_fourier = np.clip(h_fourier, 0, 1)
                # Adjust sum
                diff = N / 2.0 - np.sum(h_fourier)
                if abs(diff) > 1e-6:
                    mask = (h_fourier < 1.0) if diff > 0 else (h_fourier > 0.0)
                    if np.sum(mask) > 0:
                        h_fourier[mask] += diff / np.sum(mask)
                        h_fourier = np.clip(h_fourier, 0, 1)
                starts.append(h_fourier.copy())
    
    # Run L-BFGS-B from each start
    for i, h0 in enumerate(starts):
        if time_left() < 15:
            break
        
        try:
            result = minimize(
                objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.clip(result.x, 0, 1)
            # Enforce sum constraint more strictly
            diff = N / 2.0 - np.sum(h_opt)
            if abs(diff) > 1e-8:
                # Project to sum = N/2
                if diff > 0:
                    mask = h_opt < 1.0
                else:
                    mask = h_opt > 0.0
                if np.sum(mask) > 0:
                    h_opt[mask] += diff / np.sum(mask)
                    h_opt = np.clip(h_opt, 0, 1)
            
            val = compute_max_overlap(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Final projection to ensure sum = N/2
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-8:
        if diff > 0:
            mask = best_h < 1.0
        else:
            mask = best_h > 0.0
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
