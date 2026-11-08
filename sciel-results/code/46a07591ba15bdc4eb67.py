"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    # We aim to construct a sequence h of length N with values in [0, 1]
    # such that sum(h) = N/2 and the maximum overlap with shifted (1-h) is minimized.
    # The theoretical lower bound for the maximum overlap is N/4.
    # The constant sequence 0.5 achieves this.
    # Another class of sequences that achieves this bound (or very close to it)
    # uses values {0.25, 0.75} arranged with near-zero autocorrelation.
    # We use a deterministic pseudo-random sequence (Hadamard-like or LFSR)
    # to create structural novelty while maintaining the optimal performance.
    
    # Generate a sequence of -1 and 1 with low autocorrelation.
    # For N=100, we can use a simple LFSR or just a fixed random seed with np.
    # To be robust and fast, we use a deterministic construction.
    # Let's use a sequence based on the Legendre sequence or just random with fixed seed.
    # Since we need sum=0 for the deviations from 0.5, we need equal 0.25 and 0.75.
    
    # Create a base sequence of 0s and 1s with exactly N/2 of each.
    # We arrange them to minimize autocorrelation.
    # A simple "interleaved" pattern is bad for k=1.
    # A "Thue-Morse" like sequence or just a random permutation with fixed seed is safer.
    # Let's use a fixed random permutation of N/2 ones and N/2 zeros.
    
    rng = np.random.default_rng(42)
    mask = np.zeros(N, dtype=int)
    # Set first N/2 to 1, rest to 0
    mask[:N//2] = 1
    # Shuffle
    rng.shuffle(mask)
    
    # Map to 0.25 and 0.75
    # h_i = 0.5 + 0.25 * (2*mask_i - 1)
    # If mask=1 -> 0.5 + 0.25 = 0.75
    # If mask=0 -> 0.5 - 0.25 = 0.25
    h = 0.5 + 0.25 * (2 * mask - 1)
    
    return h

# EVOLVE-BLOCK-END
