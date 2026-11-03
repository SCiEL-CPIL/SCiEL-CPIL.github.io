"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 (a chaotic system) to generate the sign sequence.
    Different irrational initial conditions produce different ergodic trajectories.
    We evaluate the C3 bound (max absolute value of the triple autocorrelation)
    for multiple trajectories and select the one with the minimum bound.
    A fast local search is then applied to the best candidate to further minimize the bound.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients, excluding the main lobe.
        """
        # Pad to power of 2 for FFT efficiency
        # Convolution of 3 sequences of length N has length 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        
        # Identify the main lobe (global maximum) and exclude it
        # The main lobe is the largest peak. We exclude the index of the max
        # and a small neighborhood to avoid the "skirts" of the main lobe.
        abs_C3 = np.abs(C3)
        idx_max = np.argmax(abs_C3)
        
        # Exclude a window of size 3 around the max (idx_max-1, idx_max, idx_max+1)
        # This is a heuristic to remove the main lobe peak.
        # For N=400, the main lobe is narrow relative to the sidelobes.
        mask = np.ones(N_pad, dtype=bool)
        mask[idx_max] = False
        if idx_max > 0: mask[idx_max - 1] = False
        if idx_max < N_pad - 1: mask[idx_max + 1] = False
        
        # If all are masked (unlikely), return 0
        if not np.any(mask):
            return 0.0
            
        return np.max(abs_C3[mask])

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1 - x_n)
    # We try several irrational seeds to explore different chaotic trajectories.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # Simulated Annealing local search on the best candidate
    # This allows escaping local minima, which is crucial for reducing the peak autocorrelation.
    signs = best_signs
    current_bound = best_bound
    
    # Parameters for Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    alpha = 0.999 # Cooling rate
    
    import math
    import random
    import time
    
    # Set seed for reproducibility of the search process itself
    random.seed(42)
    
    # Time-limited search
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for overhead and final polish
    
    T = T_start
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random index to flip
        i = random.randint(0, N - 1)
        
        # Flip
        signs[i] *= -1.0
        new_bound = compute_c3_bound(signs)
        
        delta = new_bound - current_bound
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_bound = new_bound
        else:
            # Revert
            signs[i] *= -1.0
            
        T *= alpha
        
    # Final greedy pass to polish (coordinate descent)
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
                
    return signs
# EVOLVE-BLOCK-END
