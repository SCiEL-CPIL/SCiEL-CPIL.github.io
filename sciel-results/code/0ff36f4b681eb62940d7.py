"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    # Allow 110 seconds to be safe against the 120s limit
    deadline = start_time + 110.0
    
    # Initialize with a random signed step function
    # Using +/- 1 values
    x = np.random.choice([-1.0, 1.0], N)
    
    def compute_peak_sidelobe(x):
        # Compute aperiodic autocorrelation using FFT
        # Zero-pad to 2*N to avoid circular convolution
        size = 2 * N
        fft_x = np.fft.fft(x, size)
        power = np.abs(fft_x) ** 2
        acf = np.fft.ifft(power).real
        acf = acf[:N]
        # The peak is at lag 0. We want the max of the sidelobes (lag 1 to N-1)
        if N > 1:
            return np.max(np.abs(acf[1:]))
        return 0.0

    current_score = compute_peak_sidelobe(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local search: flip bits to reduce the peak sidelobe
    while time.time() < deadline:
        improved = False
        # Scan all positions for a flip that improves the score
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Flip bit i
            x[i] *= -1.0
            new_score = compute_peak_sidelobe(x)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                # Break to restart the scan from the beginning
                # This helps escape local minima more effectively than continuing the scan
                break
            else:
                # Revert if no improvement
                x[i] *= -1.0
        
        if not improved:
            # If no single bit flip improves the score, try a few random flips
            # to jump out of the local minimum
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = np.random.randint(N)
                x[idx] *= -1.0
                new_score = compute_peak_sidelobe(x)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                    break
                else:
                    x[idx] *= -1.0
            
            # If still stuck, we could restart with a new random sequence,
            # but keeping the best found so far is safer.
            # Given the time budget, the local search usually finds a very good solution quickly.
            
    return best_x
# EVOLVE-BLOCK-END
