import numpy as np
import time

N = 400

def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = None
    best_val = float('inf')
    
    # We will run multiple SA runs.
    # Each run starts from the current best (or random for the first one).
    
    np.random.seed(42)
    
    # Initial random start
    h = np.random.choice([-1.0, 1.0], size=N)
    current_val = _max_sidelobe(h)
    best_h = h.copy()
    best_val = current_val
    
    # SA Parameters
    # T_start should be high enough to accept most moves initially
    # Max possible delta in sidelobe? 
    # Flipping one bit changes R(k) by up to 4*1*1 = 4 for each k.
    # Max sidelobe is roughly sqrt(N) to N.
    # Let's start T at 10.0
    T_start = 10.0
    T_end = 0.01
    # Number of steps per run
    # We can fit ~100k-200k steps in 145s.
    # Let's do 5 runs of 20,000 steps each? Or adaptive based on time.
    
    run_count = 0
    max_runs = 10
    
    while time.time() < deadline and run_count < max_runs:
        run_count += 1
        T = T_start
        # Cool down factor: we want T to reach T_end after ~20000 steps
        # T * cooling^steps = T_end => cooling = (T_end/T_start)^(1/steps)
        steps_in_run = 20000
        cooling = (T_end / T_start) ** (1.0 / steps_in_run)
        
        h = best_h.copy()
        current_val = best_val
        local_best_h = h.copy()
        local_best_val = current_val
        
        no_improve_count = 0
        stagnation_limit = 2000 # If no improvement for 2000 steps, break early
        
        for step in range(steps_in_run):
            if time.time() > deadline:
                break
            
            # Pick random bit to flip
            i = np.random.randint(N)
            h[i] = -h[i]
            
            new_val = _max_sidelobe(h)
            delta = new_val - current_val
            
            # Acceptance criterion
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = new_val
                if new_val < local_best_val:
                    local_best_val = new_val
                    local_best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                h[i] = -h[i] # Revert
                no_improve_count += 1
            
            T *= cooling
            
            if no_improve_count > stagnation_limit:
                break
        
        # Update global best
        if local_best_val < best_val:
            best_val = local_best_val
            best_h = local_best_h
        else:
            # If we didn't improve, maybe we should restart from a random state 
            # or just continue from the same best with a new temperature?
            # Continuing from the same best with a new temperature is "re-annealing".
            # This helps escape local minima around the current best.
            pass
            
    return best_h
