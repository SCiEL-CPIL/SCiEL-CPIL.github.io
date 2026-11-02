import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize with random +/- 1
    # Use a fixed seed for reproducibility in case of multiple calls? 
    # No, random is fine, but let's use a good initial sequence.
    # Random is a good start.
    x = np.random.choice([1, -1], size=N)
    
    def get_max_sidelobe(x):
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        A = np.fft.ifft(X * np.conj(X)).real
        # A[0] is lag 0. A[1:N] are positive lags.
        # We want max absolute value of sidelobes.
        return np.max(np.abs(A[1:N]))
    
    current_max = get_max_sidelobe(x)
    best_x = x.copy()
    best_max = current_max
    
    # Local search
    # To make it faster, we can update autocorrelation incrementally?
    # FFT is O(N log N) ~ 400 * 9 ~ 3600 ops.
    # 10,000 iterations * 3600 = 36,000,000 ops. Fast enough in Python with NumPy?
    # NumPy FFT is C-optimized. Should be fine.
    
    # Add a bit of simulated annealing or just greedy?
    # Greedy might get stuck.
    # Let's do a few restarts.
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Use initial x
            pass
        else:
            x = np.random.choice([1, -1], size=N)
            current_max = get_max_sidelobe(x)
            if current_max < best_max:
                best_max = current_max
                best_x = x.copy()
        
        # Local search for this restart
        steps_in_restart = 0
        while time.time() < deadline and steps_in_restart < 2000:
            i = np.random.randint(N)
            old_val = x[i]
            x[i] = -old_val
            
            new_max = get_max_sidelobe(x)
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
            else:
                # Revert
                x[i] = old_val
            
            steps_in_restart += 1
            
    return best_x
