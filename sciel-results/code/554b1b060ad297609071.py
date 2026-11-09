import time
import random
import numpy as np

def hamming_weight(x):
    return x.bit_count()

def hamming_dist(a, b):
    return (a ^ b).bit_count()

# Precompute popcount for 16-bit integers
POP16 = np.array([i.bit_count() for i in range(2**16)], dtype=np.int8)

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 17
    MIN_DIST = 6
    MAX_WORDS = 2**N
    
    best_code = []
    best_size = 0
    
    # All possible words
    all_words = np.arange(MAX_WORDS, dtype=np.int32)
    
    # Run multiple greedy instances with different random orderings
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        rng = np.random.RandomState(seed)
        available = all_words.copy()
        rng.shuffle(available)
        
        # min_dist[i] is the min distance from available[i] to the current code
        # Initialize to a large number
        min_dist = np.full(len(available), 100, dtype=np.int8)
        
        code = []
        
        while len(available) > 0:
            if time.time() > DEADLINE:
                break
            
            # Find the candidate with the largest minimum distance to the code
            idx_max = np.argmax(min_dist)
            
            if min_dist[idx_max] < MIN_DIST:
                break
            
            w = available[idx_max]
            code.append(w)
            
            # Update min_dist for all remaining candidates
            # dists = popcount(available ^ w)
            diffs = available ^ w
            # Lower 16 bits popcount
            dists = POP16[diffs & 0xFFFF]
            # Add bit 16 (shifted down to bit 0)
            dists += (diffs >> 16)
            
            min_dist = np.minimum(min_dist, dists)
            
            # Remove the selected word from available and min_dist
            # Using np.delete is O(N), but N is ~131k, so it's fast enough for ~256 iterations
            available = np.delete(available, idx_max)
            min_dist = np.delete(min_dist, idx_max)
            
        if len(code) > best_size:
            best_size = len(code)
            best_code = code

    # Fallback if we didn't find anything (should not happen)
    if not best_code:
        # Use the simple RM code from the seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        best_code = words

    return best_code
