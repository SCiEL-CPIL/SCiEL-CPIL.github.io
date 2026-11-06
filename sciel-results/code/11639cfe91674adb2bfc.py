import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    
    # Start with the zero codeword
    codes = [0]
    
    # We will iterate through all possible integers
    # To make it faster, we can use a bit-mask approach or just loop.
    # 2^17 = 131072 iterations.
    # In each iteration, we check against ~250 existing codes.
    # 131072 * 250 = 32,768,000 distance checks.
    # In pure Python, this might be slow (1-2 seconds per check batch?).
    # Let's use NumPy for the distance check to vectorize it.
    
    # Store codes as a list of integers for easy addition, 
    # but for checking, convert to a NumPy array of bits?
    # Or just use a helper function that is optimized?
    
    # Actually, let's try a pure Python loop first but optimized with bit_count.
    # If it's too slow, we switch to NumPy.
    # 32 million bit_count operations in Python might take ~5-10 seconds.
    # This is within the 60s limit.
    
    # Let's try to be smarter.
    # We can maintain a NumPy array of the current codes.
    
    codes_np = np.array(codes, dtype=np.uint8)
    
    # Pre-compute bit positions? No, just use integer operations.
    
    # Let's use a hybrid: 
    # Keep a list of ints.
    # For each candidate, check against the list.
    # To speed up, we can break early if we find a conflict.
    
    # Optimization: 
    # Instead of checking all candidates 0..2^17, 
    # we can use a recursive construction or just the greedy.
    
    # Let's stick to the greedy but ensure it's fast.
    # Using a set for O(1) lookup of existence is not needed for distance.
    
    # Let's try the NumPy vectorized approach for the distance check.
    # Convert candidate to a 17-bit vector.
    # XOR with all codes (K x 17 matrix).
    # Sum along axis 1.
    # Check if min sum >= 6.
    
    # This involves creating a Kx17 array for each candidate.
    # That's 131072 * (K * 17) operations.
    # If K=250, that's 131072 * 250 * 17 ~ 550 million operations.
    # NumPy is fast, but memory allocation might be an issue.
    
    # Alternative:
    # Keep codes in a list.
    # For each candidate, loop through codes.
    # `if (cand ^ c).bit_count() < 6: break`
    # This is O(K) per candidate.
    # Total O(2^17 * K).
    # With K ~ 250, and 2^17 ~ 131k, total ops ~ 32M.
    # Python loop overhead is ~100ns per op? 32M * 100ns = 3.2 seconds.
    # This is acceptable.
    
    codes_list = [0]
    
    for cand in range(1, 1 << N):
        # Check if cand is valid
        valid = True
        for c in codes_list:
            if (cand ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            codes_list.append(cand)
            
    return codes_list
