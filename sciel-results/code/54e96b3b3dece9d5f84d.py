"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # We want to minimize an "autoconvolution upper bound".
    # A common metric is the Peak Sidelobe Level (PSL) of the aperiodic autocorrelation.
    # The seed program returns np.ones(N), which has a very high PSL (N-1).
    # We will use a random +/- 1 sequence and optimize it via local search (hill climbing)
    # to minimize the PSL.
    
    # Initialize with random signs
    # Using a fixed seed for reproducibility in evaluation if needed, 
    # but random start is usually fine. We'll use a few restarts if time permits.
    np.random.seed(42) # Fixed seed for consistency
    f = np.random.choice([-1, 1], size=N)
    
    def calculate_psl(seq):
        """
        Calculate the Peak Sidelobe Level of the aperiodic autocorrelation.
        R[t] = sum(seq[i] * seq[i+t]) for i=0..N-t-1
        We look at max |R[t]| for t > 0.
        """
        # Efficient autocorrelation using FFT or direct sum?
        # N=400 is small. Direct sum is O(N^2) = 160,000 ops. Very fast.
        # We can use np.correlate for simplicity and speed (C implementation).
        # np.correlate(seq, seq, mode='full') gives lags from -(N-1) to N-1.
        # The positive lags are in the second half.
        ac = np.correlate(seq, seq, mode='full')
        # ac[0:N-1] corresponds to negative lags, ac[N-1] is lag 0, ac[N:N-1+N] is positive lags.
        # Actually, for 'full', index N-1 is lag 0.
        # Indices N to 2N-2 are positive lags 1 to N-1.
        sidelobes = ac[N:] # This slice gives lags 1 to N-1
        return np.max(np.abs(sidelobes))
    
    # Initial cost
    current_cost = calculate_psl(f)
    
    # Hill Climbing
    # We have 120s. Each cost evaluation is ~0.1ms. 
    # 10,000 evaluations = 1s. We can do millions.
    # Let's do a simple local search.
    
    start_time = time.time()
    max_time = 100.0 # Keep buffer
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Shuffle indices to avoid bias in optimization order
        indices = np.random.permutation(N)
        
        for i in indices:
            # Flip sign
            f[i] = -f[i]
            new_cost = calculate_psl(f)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                # Revert
                f[i] = -f[i]
                
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We could restart with a new random sequence, but PSL optimization for N=400
        # often converges well with a single good random start + local search.
        # To be safe, let's do a few random restarts if we have time.
        if not improved:
            # Try a new random start if we haven't found a very good solution yet
            # Or just break. For N=400, local search from random usually gets PSL < 20.
            # Let's try one more restart if time allows.
            if current_cost > 10:
                 f = np.random.choice([-1, 1], size=N)
                 current_cost = calculate_psl(f)
            else:
                 break

    return f
# EVOLVE-BLOCK-END
