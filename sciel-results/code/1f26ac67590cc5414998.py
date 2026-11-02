import numpy as np
import time

N = 100
W = 2.0 / N

def _project_sum_box(h, target_sum):
    """Project h onto {0 <= x_i <= 1, sum(x) = target_sum} using bisection on threshold."""
    lo, hi = h.min() - 2.0, h.max() + 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h - mid, 0.0, 1.0))
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2.0
    return np.clip(h - tau, 0.0, 1.0)

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: uniform
    h = np.full(N, 0.5)
    
    # Quantization levels for discrete search
    # We use a set of values that sum to N/2 on average (0.5)
    # e.g., 0.0, 0.1, ..., 1.0
    levels = np.arange(0, 1.01, 0.1)
    
    best_h = h.copy()
    best_val = _overlap_max(h)
    current_val = best_val
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.995 # Cooling rate
    
    T = T_start
    step = 0
    max_steps = 10000
    
    # Precompute indices for random selection
    indices = np.arange(N)
    
    while step < max_steps and time.time() < deadline:
        # Pick a random index and a random level
        idx = np.random.choice(indices)
        new_level = np.random.choice(levels)
        
        h_new = h.copy()
        h_new[idx] = new_level
        
        # Project to feasible set
        h_new = _project_sum_box(h_new, N / 2.0)
        
        val_new = _overlap_max(h_new)
        
        # Acceptance criterion
        if val_new < current_val:
            h = h_new
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_h = h_new.copy()
        else:
            # Boltzmann acceptance
            delta = val_new - current_val
            if delta < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta / T)
            
            if accept:
                h = h_new
                current_val = val_new
        
        # Cool down
        T *= alpha
        
        # If T is too low, reheat or restart? 
        # For simplicity, just stop or maintain low T
        if T < T_end:
            T = T_end
            
        step += 1
        
        # Adaptive: if stuck, try a larger perturbation (change 2 indices)
        if step % 100 == 0:
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            l1, l2 = np.random.choice(levels, 2)
            h_test = h.copy()
            h_test[idx1] = l1
            h_test[idx2] = l2
            h_test = _project_sum_box(h_test, N / 2.0)
            val_test = _overlap_max(h_test)
            if val_test < current_val:
                h = h_test
                current_val = val_test
                if current_val < best_val:
                    best_val = current_val
                    best_h = h_test.copy()

    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # One final bisection projection to ensure exact sum after clipping
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
