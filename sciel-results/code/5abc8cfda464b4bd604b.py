import numpy as np
import time

N = 400
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _compute_peak_sidelobe(x):
    """
    Compute the peak sidelobe of the autocorrelation of x.
    x is a 1D array of length N.
    Returns max_{k != 0} |R[k]| where R is the aperiodic autocorrelation.
    """
    # Use FFT for efficient computation
    # Size must be at least 2*N to avoid circular convolution effects for lags up to N-1
    size = 1
    while size < 2 * N:
        size *= 2
    
    X = np.fft.rfft(x, size)
    # Autocorrelation is inverse FFT of |X|^2
    # Note: np.fft.irfft returns real values
    acf = np.fft.irfft(X * np.conj(X), size)
    
    # The aperiodic autocorrelation R[k] for k=0..N-1 is in acf[0:N]
    # R[0] is the energy (sum of squares), which is N for +-1 sequences.
    # We want the max of |R[k]| for k=1..N-1
    # Note: acf[N:] contains the negative lags, but due to symmetry for real signals,
    # R[k] = R[-k]. So checking 1..N-1 is sufficient for the peak magnitude.
    
    if N <= 1:
        return 0.0
        
    sidelobes = np.abs(acf[1:N])
    return np.max(sidelobes) if len(sidelobes) > 0 else 0.0

def _simulated_annealing(x_init, time_limit):
    """
    Perform Simulated Annealing to minimize the peak sidelobe.
    """
    start_time = time.time()
    x = x_init.astype(np.float64).copy()
    current_val = _compute_peak_sidelobe(x)
    best_x = x.copy()
    best_val = current_val
    
    # Initial temperature
    # The change in peak sidelobe from a single flip is usually small (order 1 to N)
    # Let's start with a temperature that allows some exploration
    T_start = 10.0
    T_end = 0.01
    # Number of iterations per temperature step
    # We will adjust dynamically based on time
    
    # Cooling rate
    # We want to cool from T_start to T_end over the available time.
    # Let's estimate iterations per second.
    # One iteration: 1 flip, 1 FFT eval.
    # FFT of size 1024 is very fast (~microseconds).
    # Let's aim for ~100,000 iterations total.
    
    # Dynamic loop
    it = 0
    # We'll use a fixed number of iterations per "step" of temperature
    # and adjust the temperature schedule.
    
    # A common schedule is geometric cooling: T_{k+1} = alpha * T_k
    # Let's pick alpha such that we reach T_end in roughly 100k iterations.
    # But time is the constraint.
    
    # Let's run in batches.
    batch_size = 1000
    alpha = 0.999 # Cooling factor per batch
    
    T = T_start
    rng = np.random.default_rng(42)
    
    while T > T_end:
        # Run a batch of iterations
        for _ in range(batch_size):
            if time.time() - start_time > time_limit - 2.0:
                return best_x, best_val
            
            # Flip a random bit
            i = rng.integers(0, N)
            x[i] *= -1.0
            
            new_val = _compute_peak_sidelobe(x)
            delta = new_val - current_val
            
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = x.copy()
            else:
                # Revert
                x[i] *= -1.0
                
        # Cool down
        T *= alpha
        
    return best_x, best_val

def _local_search(x, time_limit):
    """
    Hill climbing local search. Try flipping each bit and keep if it improves.
    """
    start_time = time.time()
    x = x.copy()
    current_val = _compute_peak_sidelobe(x)
    improved = True
    
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit - 2.0:
                return x, current_val
            
            x[i] *= -1.0
            new_val = _compute_peak_sidelobe(x)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                x[i] *= -1.0 # Revert
                
    return x, current_val

def construct():
    """
    Construct a signed step function (±1 sequence of length N) that minimizes
    the C3 autoconvolution upper bound.
    """
    start_time = time.time()
    
    # Initialize with random ±1 sequence
    rng = np.random.default_rng(42)
    x_init = rng.choice([-1.0, 1.0], size=N)
    
    # Phase 1: Simulated Annealing
    # Allocate ~80% of time to SA
    sa_time_limit = time_limit * 0.8
    x_sa, val_sa = _simulated_annealing(x_init, sa_time_limit)
    
    # Phase 2: Local Search Polish
    # Allocate remaining time to local search
    ls_time_limit = time_limit
    x_final, val_final = _local_search(x_sa, ls_time_limit)
    
    # Ensure we return a valid ±1 sequence
    x_final = np.sign(x_final)
    x_final[x_final == 0] = 1.0
    
    return x_final

# The evaluator calls construct()
# We don't need to define main or anything else.
