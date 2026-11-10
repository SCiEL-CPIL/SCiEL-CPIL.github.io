import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24
    
    # Build mask array using NumPy
    mask = np.zeros(max_val + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        bit = 1 << i
        mask[p::p] |= bit
    
    # For each window of 24, compute OR
    # Use sliding window with a count of how many times each bit appears
    # Actually, let's just compute OR for each window using a different approach
    
    # Precompute: for each position, we want OR of mask[i+1]..mask[i+24]
    # We can use a sliding window with counts per bit
    
    # Initialize counts for first window: positions 1001 to 1024
    # But we need windows starting at n=1000, covering n+1 to n+24
    
    # Let's compute for all n from 1000 to 1000000
    # Window for n: indices n+1 to n+24
    
    # Use a count array for each bit position
    # For the initial window n=1000: indices 1001..1024
    bit_counts = [0] * 9
    for idx in range(1001, 1025):
        m = mask[idx]
        for b in range(9):
            if m & (1 << b):
                bit_counts[b] += 1
    
    # OR mask for current window
    current_or = 0
    for b in range(9):
        if bit_counts[b] > 0:
            current_or |= (1 << b)
    
    best_n = 1000
    best_count = bin(current_or).count('1')
    
    # Slide the window
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove index n (which was n+1 for previous n-1... wait let me re-think)
        # For n-1: window is (n-1)+1 to (n-1)+24 = n to n+23
        # For n: window is n+1 to n+24
        # So we remove index n and add index n+24
        
        # Remove mask[n]
        m_remove = mask[n]
        for b in range(9):
            if m_remove & (1 << b):
                bit_counts[b] -= 1
        
        # Add mask[n+24]
        m_add = mask[n + 24]
        for b in range(9):
            if m_add & (1 << b):
                bit_counts[b] += 1
        
        # Update OR
        current_or = 0
        for b in range(9):
            if bit_counts[b] > 0:
                current_or |= (1 << b)
        
        count = bin(current_or).count('1')
        if count < best_count:
            best_count = count
            best_n = n
            if best_count <= 1:
                break
    
    return best_n
