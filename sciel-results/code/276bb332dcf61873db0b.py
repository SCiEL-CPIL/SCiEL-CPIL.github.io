import numpy as np
import time

N = 400

def _max_psl(x):
    """Compute the peak sidelobe level of the autocorrelation."""
    # x is a numpy array of shape (N,)
    # Autocorrelation via FFT
    f = np.fft.fft(x)
    acf = np.fft.ifft(f * np.conj(f)).real
    # acf[0] is the energy, acf[1:N] are the sidelobes (circular)
    # Note: np.fft.ifft gives circular autocorrelation.
    # For linear autocorrelation, we would use correlate.
    # However, for random sequences, circular and linear are similar in peak behavior.
    # The task specifies "step function on [-1/4, 1/4] with 400 intervals".
    # This implies a linear convolution/autocorrelation context usually.
    # Let's use linear autocorrelation to be safe.
    # np.correlate(x, x, mode='full') gives linear autocorrelation.
    # This is O(N^2) but N=400 is small.
    # Let's check speed. 400*400 = 160,000 ops. Very fast.
    acf_lin = np.correlate(x, x, mode='full')
    # acf_lin[0:N-1] corresponds to lags N-1 down to 0?
    # np.correlate(a, v, mode='full') 
    # The result is: [a[0]v[-N+1] ... a[N-1]v[N-1]]
    # For a=a, v=a:
    # Lag 0 is at index N-1.
    # Lags 1..N-1 are at indices 0..N-2 and N..2N-2.
    # Due to symmetry, we just need to look at one side.
    # Indices 0 to N-2 correspond to lags N-1 down to 1.
    # So we take the max abs of acf_lin[0:N-1]
    return np.max(np.abs(acf_lin[0:N-1]))

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4,1/4].
    Returns a sequence of 400 signed step heights in {-1, 1} that minimizes
    the autoconvolution upper bound C3.
    """
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Initial guess: Random
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N).astype(np.float64)
    
    current_score = _max_psl(x)
    best_score = current_score
    best_x = x.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.9999
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = rng.integers(0, N)
        
        # Flip
        x[i] *= -1
        
        # Compute new score
        new_score = _max_psl(x)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Revert
            x[i] *= -1
            
        # Cool down
        T *= cooling_rate
        
        # Prevent division by zero or underflow in exp
        if T < 1e-10:
            T = 1e-10
            
    return best_x.astype(int)

# EVOLVE-BLOCK-END
