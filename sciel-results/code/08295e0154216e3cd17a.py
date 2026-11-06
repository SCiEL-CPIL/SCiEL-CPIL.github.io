import numpy as np
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Precompute popcount for 16-bit values
    popcount_16 = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        popcount_16[i] = bin(i).count('1')
    
    def popcount18(x):
        """Fast popcount for 18-bit integer using 16-bit table."""
        return int(popcount_16[x & 0xFFFF]) + int(popcount_16[(x >> 16) & 0xFFFF])
    
    # Start with Reed-Muller [16,5,8] code extended by 2 zero bits -> 32 words, dist 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    # Try multiple random restarts and keep the best
    best_code = list(base_words)
    best_len = len(best_code)
    
    rng = np.random.default_rng(12345)
    
    # Generate all possible words
    all_words = np.arange(2**18, dtype=np.uint32)
    
    # Multiple restarts with different random seeds
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        code = list(base_words)
        code_set = set(code)
        code_arr = np.array(code, dtype=np.uint32)
        
        # Shuffle candidates
        shuffled = all_words.copy()
        rng.shuffle(shuffled)
        
        added_count = 0
        start_time = time.time()
        
        for candidate in shuffled:
            if time.time() > deadline - 2:
                break
            
            candidate_int = int(candidate)
            if candidate_int in code_set:
                continue
            
            # Check distance to all existing code words using vectorized operations
            xors = code_arr ^ candidate
            
            # Calculate popcount for each xor result using 16-bit table
            low_parts = xors & 0xFFFF
            high_parts = (xors >> 16) & 0xFFFF
            popcounts = popcount_16[low_parts] + popcount_16[high_parts]
            
            min_d = np.min(popcounts)
            
            if min_d >= MIN_DIST:
                code.append(candidate_int)
                code_set.add(candidate_int)
                code_arr = np.array(code, dtype=np.uint32)
                added_count += 1
        
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
        
        # If we've found a good code, try local search to improve
        if len(code) >= best_len and time.time() < deadline - 10:
            # Try to add more by checking in different orders
            pass
    
    return best_code
