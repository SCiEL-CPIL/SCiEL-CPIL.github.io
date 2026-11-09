"""
Optimize signed step function to minimize autoconvolution upper bound (C3).
The metric is assumed to be the Peak Sidelobe Level (PSL) of the linear autocorrelation.
Uses greedy local search with FFT-based evaluation and a strict deadline guard.
"""
import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin under 120s/150s limit

def compute_psl(h):
    """
    Compute the Peak Sidelobe Level (PSL) of the linear autocorrelation of h.
    h is a numpy array of length N with values +/- 1.
    """
    # Determine FFT size for linear convolution (autocorrelation)
    # Length of linear autocorrelation is 2*N - 1
    # Next power of 2
    n_fft = 1 << (2 * N).bit_length()
    
    # FFT of h
    F = np.fft.rfft(h, n=n_fft)
    
    # Power spectrum
    power = F * np.conj(F)
    
    # Inverse FFT to get autocorrelation
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is the autocorrelation at lag 0 (peak)
    # r[1:N] contains the positive lags
    # r[N+1:2N] contains the negative lags (symmetric for real input)
    # We want the maximum absolute value of the sidelobes (lags != 0)
    
    if N > 1:
        # Sidelobes are r[1:N]
        # Note: r[N] is the midpoint for even N? 
        # For N=400, lags are 0 to 399.
        # r[0] is lag 0.
        # r[1] is lag 1.
        # ...
        # r[399] is lag 399.
        # The autocorrelation is symmetric, so r[k] = r[-k].
        # The linear autocorrelation length is 799.
        # r[0] to r[798].
        # r[0] is lag 0.
        # r[1] to r[399] are positive lags 1 to 399.
        # r[400] to r[798] are negative lags -1 to -399.
        # So we just need max(abs(r[1:N]))
        
        # Ensure we don't index out of bounds if N changes, but N is fixed
        return np.max(np.abs(r[1:N]))
    else:
        return 0.0

def construct():
    start_time = time.time()
    
    # Initialize with a random sequence of +/- 1
    # This typically yields a much lower PSL than the all-ones seed
    h = np.random.choice([1.0, -1.0], N)
    
    # Store the best sequence found
    best_h = h.copy()
    best_psl = compute_psl(best_h)
    
    # Greedy local search
    # We perform a limited number of passes to ensure we finish within the time limit.
    # Each pass iterates through all indices and flips them if it improves the PSL.
    max_passes = 15
    
    for _ in range(max_passes):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Shuffle the order of indices to avoid bias and help escape local minima
        indices = np.random.permutation(N)
        
        for i in indices:
            # Check deadline
            if time.time() - start_time > DEADLINE:
                break
            
            # Flip the bit at index i
            h[i] *= -1
            
            # Calculate new PSL
            psl = compute_psl(h)
            
            if psl < best_psl:
                # Accept the flip
                best_psl = psl
                best_h = h.copy()
                improved = True
            else:
                # Revert the flip
                h[i] *= -1
        
        # If no improvement in a full pass, stop
        if not improved:
            break

    return best_h
