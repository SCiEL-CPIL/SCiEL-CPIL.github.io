import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    total_words = 1 << N
    
    # Precompute 8-bit popcount lookup table for vectorized popcount
    pop8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    
    def popcount_vec(arr):
        """Vectorized popcount for a numpy uint32 array."""
        a = arr.astype(np.uint64)
        return (pop8[(a & 0xFF).astype(np.intp)] +
                pop8[((a >> 8) & 0xFF).astype(np.intp)] +
                pop8[((a >> 16) & 0xFF).astype(np.intp)] +
                pop8[((a >> 24) & 0xFF).astype(np.intp)]).astype(np.uint8)
    
    best_words = []
    max_restarts = 5000
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate candidate order
        if i == 0:
            # Deterministic: sort by popcount (lighter words first)
            all_w = np.arange(total_words, dtype=np.uint32)
            pc = popcount_vec(all_w)
            order = np.argsort(pc, kind='stable')
        elif i % 2 == 1:
            # Random shuffle
            order = np.random.permutation(total_words)
        else:
            # Weighted random: prefer lower-weight words
            all_w = np.arange(total_words, dtype=np.uint32)
            pc = popcount_vec(all_w).astype(np.float32)
            w = 1.0 / (1.0 + pc)
            w /= w.sum()
            order = np.random.choice(total_words, size=total_words, replace=False, p=w)
        
        words = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            c = int(idx)
            
            if not words:
                words.append(c)
                continue
            
            # Vectorized minimum distance check against all existing words
            existing = np.array(words, dtype=np.uint32)
            diffs = existing ^ np.uint32(c)
            min_dist = int(popcount_vec(diffs).min())
            
            if min_dist >= MIN_DIST:
                words.append(c)
        
        if len(words) > len(best_words):
            best_words = words[:]
            if len(best_words) >= 1024:
                break
    
    return best_words
