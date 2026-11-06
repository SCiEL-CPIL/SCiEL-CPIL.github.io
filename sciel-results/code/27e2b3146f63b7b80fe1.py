import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    DEADLINE = time.time() + 85
    
    pc = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    
    def popcount16(x):
        lo = (x & 0xFF).astype(np.uint8)
        hi = ((x >> 8) & 0xFF).astype(np.uint8)
        return pc[lo] + pc[hi]
    
    # Phase 1: Greedy construction
    code = np.array([0], dtype=np.uint16)
    
    all_words = np.arange(1, 1 << N, dtype=np.uint16)
    weights = popcount16(all_words)
    
    # Order by weight: 8,9,7,10,6,11,5,12,4,13,3,14,2,15,1,16
    order_idx = []
    for w in [8, 9, 7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 2, 15, 1, 16]:
        idx = np.where(weights == w)[0]
        if len(idx) > 0:
            order_idx.append(idx)
    
    if order_idx:
        ordered_idx = np.concatenate(order_idx)
    else:
        ordered_idx = np.arange(len(all_words))
    
    for idx in ordered_idx:
        if time.time() > DEADLINE:
            break
        w = all_words[idx]
        dists = popcount16(code ^ w)
        if np.min(dists) >= MIN_DIST:
            code = np.append(code, w)
    
    # Phase 2: Local search - try to add more words
    # Use a queue of words to try
    rng = np.random.default_rng(123)
    
    # Try many random words, add if compatible
    attempts = 0
    max_attempts = 50000
    while attempts < max_attempts:
        if time.time() > DEADLINE:
            break
        attempts += 1
        
        w = int(rng.integers(1, 1 << N))
        if w in code:
            continue
        
        dists = popcount16(code ^ w)
        if np.min(dists) >= MIN_DIST:
            code = np.append(code, w)
    
    return code.tolist()
