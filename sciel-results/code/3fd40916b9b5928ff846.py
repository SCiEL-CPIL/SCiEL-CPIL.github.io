import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Leave margin for overhead
    
    # Start with zero word
    code = np.array([0], dtype=np.int32)
    
    # Precompute all 2^17 words as a numpy array
    all_words = np.arange(2**N, dtype=np.int32)
    
    # We'll process words in a specific order: by weight, balanced first
    # Compute weights for all words
    # This is expensive but doable
    # Actually, let's just shuffle and try, with a time limit
    
    # Better: use a greedy approach where we try to add words
    # We'll maintain a bitset of which words are in the code
    
    # For efficiency, let's use a different strategy:
    # Start with a good seed (e.g., the RM code), then extend greedily
    
    # First, build the RM(1,4) code (32 words)
    rm_code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_code.append(word)
    
    code = list(rm_code)
    used = set(code)
    
    # Now try to extend: iterate through all possible words, add if compatible
    # Process in order of weight for better results
    # Precompute weight for each word? Too expensive.
    
    # Just iterate through all words and check
    # To speed up, use early termination
    
    for word in range(2**N):
        if time.time() - start_time > time_limit:
            break
        if word in used:
            continue
        # Check min distance to all existing codewords
        min_dist = N + 1
        for existing in code:
            d = (word ^ existing).bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < MIN_DIST:
                    break
        if min_dist >= MIN_DIST:
            code.append(word)
            used.add(word)
    
    return code
