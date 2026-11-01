import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random signs
    # Using a fixed seed for reproducibility in local tests, but random is fine for evaluator
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N)
    
    # Helper to compute autocorrelation
    def get_autocorr(x):
        # Compute autocorrelation via FFT
        n = len(x)
        f = np.fft.fft(x, n=2*n)
        r = np.fft.ifft(f * np.conj(f))
        # Take the real part and the correct slice
        # The autocorrelation of a real signal is symmetric.
        # We want lags 0 to n-1.
        # ifft gives circular correlation. For zero-padded, it's linear.
        r = r[:n].real
        return r
    
    # Initial autocorrelation
    R = get_autocorr(h)
    
    # Objective: Max absolute autocorrelation at non-zero lags
    # Exclude lag 0 (index 0) as it is N for +/-1 signals
    current_max = np.max(np.abs(R[1:]))
    
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Flip bit i
            h[i] *= -1
            
            # Compute new autocorrelation
            R_new = get_autocorr(h)
            new_max = np.max(np.abs(R_new[1:]))
            
            if new_max < current_max:
                current_max = new_max
                R = R_new
                improved = True
            else:
                # Revert
                h[i] *= -1
                
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We could add a perturbation step, but let's keep it simple first.
        # If we want to escape local minima, we can randomize a few bits.
        if not improved:
            # Try a few random restarts or perturbations if time allows
            if time.time() < deadline - 10:
                # Perturb a small number of random bits
                num_perturb = 5
                indices = rng.choice(N, size=num_perturb, replace=False)
                for idx in indices:
                    h[idx] *= -1
                R = get_autocorr(h)
                current_max = np.max(np.abs(R[1:]))
                improved = True # Force another pass to see if perturbation helps
                
    # Final check
    R_final = get_autocorr(h)
    # The score is seed_value / raw_value.
    # Seed is ones. R_seed[1:] is [399, 398, ..., 1]. Max is 399.
    # If the metric is max sidelobe, raw_value is current_max.
    # We return the heights. The evaluator computes the score.
    
    return h
