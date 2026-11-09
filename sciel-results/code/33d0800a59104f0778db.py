import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial random sequence
    h = np.random.choice([-1, 1], size=N)
    
    # Function to compute max non-zero lag autocorrelation
    def get_max_sidelobe(h):
        # FFT based autocorrelation
        # Pad to 2N to avoid circular convolution artifacts
        n_fft = 2 * N
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        r = np.fft.irfft(power, n=n_fft)
        # r[0] is lag 0 (sum of squares = N)
        # r[1] to r[N-1] are positive lags
        # r[n_fft - 1] ... r[n_fft - (N-1)] are negative lags
        # Due to symmetry for real h, max positive lag = max negative lag
        # So we just look at r[1:N]
        # Note: r[N] is lag N, which is 0 for length N sequence in 2N FFT?
        # Actually, linear autocorrelation of length N has lags -N+1 to N-1.
        # r[0] is lag 0.
        # r[1] is lag 1.
        # ...
        # r[N-1] is lag N-1.
        # The values at r[N] to r[2N-1] correspond to lags -N to -1?
        # Let's check.
        # Circular correlation of length 2N.
        # The linear autocorrelation is embedded in the first N and last N-1?
        # Actually, for real input, r[k] = r[n_fft-k].
        # So max(r[1:N]) is sufficient.
        
        # Exclude lag 0
        sidelobes = r[1:N]
        return np.max(np.abs(sidelobes))
    
    current_score = get_max_sidelobe(h)
    best_h = h.copy()
    best_score = current_score
    
    # Hill Climbing
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Randomize order to escape local minima? Or just iterate?
        # Let's iterate through all indices
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            
            # Try flipping h[i]
            old_val = h[i]
            h[i] = -old_val
            
            new_score = get_max_sidelobe(h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                # Revert if not better
                h[i] = old_val
        
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We could try a few random restarts or simulated annealing steps.
        # Given time constraints, simple hill climbing might be enough for a big jump from 399 to ~20.
        
    return best_h
