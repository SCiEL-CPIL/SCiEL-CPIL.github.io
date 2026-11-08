"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_c3(x):
    """
    Compute the C3 upper bound (max absolute autocorrelation sidelobe).
    Excludes k=0 to allow for non-trivial optimization of binary sequences.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    xf = np.fft.rfft(x, n=n_fft)
    # Linear autocorrelation
    # conv[k] = sum_i x[i] x[i+k]
    # Using FFT: irfft(fft(x)^2) gives circular correlation if not padded.
    # With padding, it gives linear correlation in the first 2N-1 bins.
    # We want k=1 to N-1.
    # Note: np.fft.irfft returns real values.
    conv = np.fft.irfft(xf * xf, n=n_fft)
    # The linear autocorrelation lags are 0 to N-1 in the first N bins.
    # We exclude 0.
    return np.max(np.abs(conv[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    np.random.seed(42)
    
    # Simulated Annealing Parameters
    T_start = 50.0
    T_end = 0.1
    # We don't know the total steps, so we use a time-based cooling or fixed steps.
    # Let's estimate steps. 1 step ~ 50us (FFT + logic).
    # 115s / 50us = 2.3 million steps.
    # Let's just run until deadline.
    
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _compute_c3(best_x)
    
    # Multiple restarts with annealing
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current state
        if r == 0:
            x = best_x.copy()
            current_score = best_score
        else:
            x = np.random.choice([-1.0, 1.0], size=N)
            current_score = _compute_c3(x)
            
        # Annealing schedule for this restart
        # We will run for time_per_restart
        t_start = time.time()
        t_end = t_start + time_per_restart - 1.0 # small buffer
        
        # Geometric cooling
        # T(t) = T_start * (T_end/T_start)^(t/total_time)
        # Since we don't know exact steps, we'll just decrement T linearly or geometrically per step.
        # Let's assume ~1000 steps per second? No, 1 FFT is ~10-20us. So ~50k-100k steps per second.
        # 115s -> ~5M steps.
        # Let's set T to cool over ~100k steps.
        
        T = T_start
        # Cooling factor per step
        # If we run for ~100k steps, alpha = (T_end/T_start)^(1/100000)
        alpha = (T_end / T_start) ** (1.0 / 100000.0)
        
        while time.time() < t_end:
            # Pick random index to flip
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            new_score = _compute_c3(x)
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                # Revert
                x[idx] = old_val
                
            T *= alpha
            if T < T_end:
                T = T_end
                
    return best_x

# EVOLVE-BLOCK-END
