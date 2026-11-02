"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initialize with a random sequence
    # Using a fixed seed for reproducibility in case we don't improve much
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    # Function to compute autocorrelation and objective
    # Objective: Sum of squares of non-zero lag autocorrelations
    # This minimizes the "energy" of the sidelobes.
    
    def get_objective(x):
        # Use FFT for fast autocorrelation
        # x is real, so we can use np.fft.fft
        # Autocorrelation R(k) = IDFT(|DFT(x)|^2)
        # We only care about k != 0
        fft_x = np.fft.fft(x)
        power = np.abs(fft_x)**2
        acf = np.fft.ifft(power).real
        # The zero lag is acf[0], which is sum(x^2) = N.
        # We want to minimize the sum of squares of the rest.
        # Note: acf is symmetric. acf[k] = acf[N-k].
        # The non-zero lags are acf[1:N].
        return np.sum(acf[1:]**2)
    
    current_obj = get_objective(x)
    best_x = x.copy()
    best_obj = current_obj
    
    # Local search: try flipping bits
    # To speed up, we can use a heuristic: flipping a bit at index i changes the autocorrelation.
    # Delta in objective can be computed efficiently.
    # R_new(k) = R_old(k) - 2*x_i*(x_{i-k} + x_{i+k}) + 2 (if k != 0) ... wait, this is complex.
    # Simpler: Just recompute if N is small enough? N=400 is small.
    # FFT is O(N log N). 400 * log(400) is very fast.
    # We can do many iterations.
    
    num_iterations = 0
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.integers(0, N)
        old_val = x[i]
        x[i] = -x[i]
        
        new_obj = get_objective(x)
        
        if new_obj < current_obj:
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()
        else:
            # Revert
            x[i] = old_val
            
        num_iterations += 1
        # Add a small pause to allow for time check? No, tight loop is better.
        
        # Check time every 1000 iterations to avoid overhead of time.time()
        if num_iterations % 1000 == 0:
            if time.time() > deadline - 5.0:
                break

    return best_x
# EVOLVE-BLOCK-END
