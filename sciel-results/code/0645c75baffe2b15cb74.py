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
    
    # Pre-allocate array for code words (max possible is 2^18)
    max_code_size = 2**18
    code_arr = np.zeros(max_code_size, dtype=np.uint32)
    code_arr[:len(code)] = np.array(code, dtype=np.uint32)
    code_len = len(code)
    
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
        xors = code_arr[:code_len] ^ candidate
        
        # Calculate popcount for each xor result using fast method
        # Use the standard bit-manipulation trick for popcount
        # For 32-bit integers, we can use multiple shifts and masks
        
        # Fast popcount using numpy:
        # x = x - ((x >> 1) & 0x55555555)
        # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        # x = (x + (x >> 4)) & 0x0F0F0F0F
        # x = (x * 0x01010101) >> 24
        
        temp = xors.copy()
        temp -= (temp >> 1) & np.uint32(0x55555555)
        temp = (temp & np.uint32(0x33333333)) + ((temp >> 2) & np.uint32(0x33333333))
        temp = (temp + (temp >> 4)) & np.uint32(0x0F0F0F0F)
        popcounts = (temp * np.uint32(0x01010101)) >> np.uint32(24)
        
        min_d = np.min(popcounts)
        
        if min_d >= MIN_DIST:
            code.append(candidate_int)
            code_set.add(candidate_int)
            code_arr[code_len] = candidate_int
            code_len += 1
            added_count += 1
            
    return code
