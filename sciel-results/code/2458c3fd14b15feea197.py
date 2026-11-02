"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


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
        of the third-order autocorrelation coefficients.
        """
        # Pad to power of 2 for FFT efficiency
        # Convolution of 3 sequences of length N has length 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def generate_logistic_signs(seed, n_steps=N):
        """Generate sign sequence from logistic map."""
        x = seed
        signs = np.zeros(n_steps)
        for i in range(n_steps):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs

    # Generate candidates using Logistic Map with diverse irrational seeds
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        np.sqrt(5) % 1.0,
        np.sqrt(13) % 1.0,
        np.sqrt(17) % 1.0,
        np.sqrt(29) % 1.0,
        np.sqrt(41) % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        0.5 + (1/np.sqrt(3)) % 1.0,
        0.5 + (1/np.sqrt(7)) % 1.0,
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        signs = generate_logistic_signs(seed)
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # Local search with multiple restarts to escape local minima
    signs = best_signs.copy()
    current_bound = best_bound
    
    # Precompute FFT for efficient updates
    def get_fft(s):
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        return np.fft.fft(s_pad)
    
    def compute_bound_from_fft(S):
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))
    
    # Greedy local search with multiple passes
    max_rounds = 50
    no_improvement_count = 0
    
    for round_num in range(max_rounds):
        improved = False
        S = get_fft(signs)
        
        for i in range(N):
            # Flip bit i
            signs[i] *= -1.0
            S = get_fft(signs)
            new_bound = compute_bound_from_fft(S)
            
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                # Revert
                signs[i] *= -1.0
                S = get_fft(signs)
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 3:
                break
        else:
            no_improvement_count = 0
    
    return signs
# EVOLVE-BLOCK-END
