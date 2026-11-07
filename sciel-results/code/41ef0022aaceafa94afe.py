import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    N = 10000000

    # 1. Compute sigma(n) for all n up to N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve method: for each i, add i to all multiples
    # This is O(N log N)
    for i in range(1, N + 1):
        if time.time() > deadline - 5.0:
            # Fallback if sieve takes too long (unlikely for 10^7 with numpy)
            return [11, 6]
        sigma[i::i] += i

    # 2. Search for pairs (a, b) such that sigma(a) == sigma(b) and a/b ~ 1.5
    # We iterate b and check a near 1.5*b
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # To speed up the Python loop, we can convert sigma to a list?
    # No, numpy array indexing is fast enough, but Python loop overhead is significant.
    # Let's try to vectorize the check?
    # It's hard to vectorize "check neighbors of 1.5*b" easily without creating large arrays.
    # Let's stick to the loop but optimize it.
    
    # We can skip b where 1.5*b > N + margin
    max_b = int(N * 2 / 3)
    
    # Pre-extract sigma to a list for faster access in Python loop?
    # np.array indexing is faster than list indexing for random access?
    # Actually, list of Python ints might be faster than np.int64 conversion?
    # Let's try keeping it as numpy and see.
    
    for b in range(1, max_b + 1):
        if time.time() > deadline - 2.0:
            break
            
        sb = sigma[b]
        
        # Target a is 1.5 * b
        # We check a in [1.5*b - K, 1.5*b + K]
        # K=10 is small, but might miss if the "closest" sigma match is further?
        # No, we want to MINIMIZE |a/b - 1.5|.
        # The closest a to 1.5*b is the best candidate for that b.
        # If sigma(a) != sigma(b) for the closest a, we check the next closest.
        # So checking a small window is correct for finding the global min error,
        # provided the window is large enough to contain the optimal a for some b.
        # The error is |2a - 3b| / 2b.
        # If we check a range of size 2K, we cover errors up to K/b.
        # For large b, K/b is tiny.
        # Is it possible the optimal pair has a large gap?
        # Unlikely. The density of numbers with same sigma is low.
        # But we are searching for ANY pair.
        # If we miss the optimal pair because it's outside the window, we might return a suboptimal one.
        # However, the previous code (which timed out) was exhaustive.
        # This new code is heuristic.
        # To be safe, let's increase K slightly or be smarter.
        
        # Actually, if we just want to beat the seed, a small window might suffice.
        # But to be robust, let's check a bit wider.
        # Let's check a in [round(1.5*b) - 50, round(1.5*b) + 50]
        
        target_a = 1.5 * b
        a_center = int(round(target_a))
        
        # Check range
        start_a = max(1, a_center - 50)
        end_a = min(N, a_center + 50)
        
        for a in range(start_a, end_a + 1):
            if a == b:
                continue
            if sigma[a] == sb:
                # Calculate error exactly
                # |a/b - 3/2| = |2a - 3b| / 2b
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue # Invalid
                error = diff / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b

    if best_error == float('inf'):
        return [11, 6]
    
    return [best_a, best_b]
