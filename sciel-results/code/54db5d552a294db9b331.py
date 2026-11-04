import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals using the Thue-Morse sequence.
    The Thue-Morse sequence is a deterministic low-autocorrelation sequence that
    minimizes the magnitude of autocorrelation coefficients, thereby improving
    the upper bound on the third autocorrelation coefficient (C3).
    """
    # Generate Thue-Morse sequence for indices 0 to N-1
    # t_n = popcount(n) % 2
    # Map 0 to 1.0 and 1 to -1.0
    indices = np.arange(N)
    # Compute popcount efficiently using bit manipulation or lookup
    # For N=400, a simple loop or vectorized bit-count is fast enough.
    # numpy doesn't have a direct popcount for integers, so we use a bit trick.
    
    # Method: Count bits set in each integer
    # We can use a lookup table for 8-bit chunks or just use a loop since N is small.
    # Or use a trick: x - ((x >> 1) & 0b01010101) - ((x >> 2) & 0b00110011) ... 
    # Simpler: use a precomputed table for 0..255 and combine.
    
    # Even simpler for N=400:
    signs = np.ones(N)
    for i in range(N):
        # Count set bits in i
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        if c % 2 == 1:
            signs[i] = -1.0
    return signs
# EVOLVE-BLOCK-END
