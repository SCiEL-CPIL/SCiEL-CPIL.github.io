"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # Start with the Reed-Muller construction (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    word_set = set(words)
    word_list = words
    
    # Greedy random addition: try to add random codewords that maintain min distance
    # Use numpy for vectorized distance checking when possible
    
    rng = np.random.default_rng(42)
    added = 0
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        batch_size = 1000
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in word_set:
                continue
            
            # Check distance against all existing words
            valid = True
            for w in word_list:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                word_list.append(cand)
                word_set.add(cand)
                added += 1
    
    return word_list
# EVOLVE-BLOCK-END
