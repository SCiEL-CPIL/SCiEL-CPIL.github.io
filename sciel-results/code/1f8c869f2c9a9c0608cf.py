"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 to generate the sign sequence.
    We employ a randomized local search to minimize the C3 bound.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1.0 - x_n)
    # Using several irrational seeds to ensure diverse ergodic trajectories.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        1.0 / np.sqrt(2) % 1.0,
        np.sqrt(13) % 1.0,
        (1 + np.sqrt(17)) / 2 % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    # Time limit management
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer for overhead
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
            
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        # Local Search: Randomized Hill Climbing
        # We perform multiple passes, flipping bits in random order.
        # If a flip improves the bound, we keep it.
        current_signs = signs.copy()
        current_bound = compute_c3_bound(current_signs)
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_signs = current_signs.copy()
            
        # Run local search on this candidate
        # Max iterations per candidate
        max_iter_per_candidate = 2000
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter_per_candidate:
            if time.time() - start_time > time_limit:
                break
            improved = False
            # Random permutation of indices
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                # Flip bit
                current_signs[i] *= -1.0
                new_bound = compute_c3_bound(current_signs)
                
                if new_bound < current_bound:
                    current_bound = new_bound
                    improved = True
                    if current_bound < best_bound:
                        best_bound = current_bound
                        best_signs = current_signs.copy()
                else:
                    # Revert flip
                    current_signs[i] *= -1.0
                    
            iterations += 1
            
    return best_signs
# EVOLVE-BLOCK-END
