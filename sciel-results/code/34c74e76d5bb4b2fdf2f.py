import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    Uses Simulated Annealing to escape local minima.
    """
    np.random.seed(42)
    
    # Initialize with random signs
    f = np.random.choice([-1, 1], N)
    
    def get_peak_sidelobe(f):
        """
        Computes the peak absolute value of the non-zero lag autocorrelation.
        """
        # Zero-pad to 2*N to avoid circular convolution artifacts
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        
        # Extract non-zero lags
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        
        if len(pos_lags) == 0 and len(neg_lags) == 0:
            return 0.0
            
        return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))

    current_peak = get_peak_sidelobe(f)
    best_f = f.copy()
    best_peak = current_peak
    
    # Simulated Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    cooling_rate = 0.999
    max_time = 100.0 # seconds
    start_time = time.time()
    
    temp = start_temp
    
    while temp > end_temp and (time.time() - start_time) < max_time:
        # Pick a random index to flip
        i = np.random.randint(N)
        old_val = f[i]
        f[i] = -old_val
        
        new_peak = get_peak_sidelobe(f)
        
        delta = new_peak - current_peak
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
        else:
            # Revert
            f[i] = old_val
            
        temp *= cooling_rate
        
    return best_f
# EVOLVE-BLOCK-END
