import numpy as np

N = 100

def construct():
    """
    Construct the sequence of step heights using a domain-wall approach.
    We model the sequence as two blocks of 1s (length N/4 each) separated by a gap,
    and the rest as 0s. We optimize the gap size to minimize the maximum overlap.
    """
    # Target sum is N/2. 
    # Use two blocks of 1s of length N/4 = 25.
    L = N // 4
    best_h = None
    best_score = float('inf')
    
    # We can shift the blocks or change the gap.
    # Due to symmetry, we can fix the first block at [0, L) and vary the start of the second block.
    # Let second block start at s.
    # To avoid overlap and keep sum correct, s must be >= L.
    # The sequence is circular.
    
    for s in range(L, N - L + 1):
        h = np.zeros(N)
        h[0:L] = 1.0
        # Add second block
        # Handle wrap around
        idx = np.arange(s, s + L) % N
        h[idx] = 1.0
        
        # Check sum (should be 2*L = N/2)
        if np.sum(h) != N / 2.0:
            continue
            
        # Calculate max overlap
        # C_k = sum h_i (1 - h_{i+k})
        # We can compute this efficiently
        max_c = 0
        for k in range(1, N // 2 + 1):
            # h_shifted = np.roll(h, -k)
            # c = np.sum(h * (1 - h_shifted))
            # Using np.roll is slow for N=100 in a loop 50 times, but acceptable.
            h_shifted = np.roll(h, -k)
            c = np.sum(h * (1 - h_shifted))
            if c > max_c:
                max_c = c
        
        if max_c < best_score:
            best_score = max_c
            best_h = h.copy()
            
    # If no valid h found (shouldn't happen), fallback to cosine
    if best_h is None:
        i = np.arange(N)
        best_h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        
    return best_h
