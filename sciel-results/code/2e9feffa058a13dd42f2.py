import numpy as np
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Start with Reed-Muller [16,5,8] code extended by 2 zero bits -> 32 words, dist 8
    # This gives us a strong seed
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    code = list(base_words)
    code_set = set(code)
    code_arr = np.array(code, dtype=np.uint32)
    
    # Greedy addition: try to add more words
    # Use numpy for fast distance checking
    
    # Strategy: iterate through all candidates in a random order
    # Check each candidate against the entire current code using vectorized operations
    
    rng = np.random.default_rng(42)
    
    # Generate all possible words and shuffle
    all_words = np.arange(2**18, dtype=np.uint32)
    rng.shuffle(all_words)
    
    # We iterate through the shuffled array
    # We don't filter initially to avoid large list creation, just check membership
    
    added_count = 0
    start_time = time.time()
    
    for candidate in all_words:
        if time.time() > deadline:
            break
        
        candidate_int = int(candidate)
        if candidate_int in code_set:
            continue
        
        # Check distance to all existing code words
        # Vectorized XOR
        xors = code_arr ^ candidate
        
        # Calculate popcount for each xor result
        # Using the bit-shifting loop which is fast in NumPy
        popcounts = np.zeros_like(xors, dtype=np.int32)
        for i in range(N):
            popcounts += (xors >> i) & 1
            
        min_d = np.min(popcounts)
        
        if min_d >= MIN_DIST:
            code.append(candidate_int)
            code_set.add(candidate_int)
            # Update code_arr
            code_arr = np.array(code, dtype=np.uint32)
            added_count += 1
            
    return code
