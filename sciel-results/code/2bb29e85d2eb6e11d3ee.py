import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Create a triangular shape centered at 0
    # Width N, peak at center
    h = np.zeros(N)
    center = N // 2
    # Triangle: 1, 2, ..., center, ..., 2, 1
    # Indices: 0 to N-1
    # Height at i: min(i, N-1-i, center) + 1 ?
    # Let's just make it symmetric.
    # If N is even, center is N/2 - 0.5?
    # N=600. Center index 299.5.
    # Let's use 1 to 300 and back.
    for i in range(N):
        # Distance from center
        dist = abs(i - (N - 1) / 2.0)
        # Max height is N/2
        h[i] = max(0, N / 2.0 - dist)
        
    # Ensure non-negative
    h[h < 0] = 0
    
    return h.tolist()
# EVOLVE-BLOCK-END
