import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Fast popcount for 18-bit values using 3x6-bit lookup
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
    
    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)
    
    # Sort candidates by weight (excluding 0)
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))
    
    best_words = [0]
    best_len = 1
    
    rng = np.random.default_rng(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        if attempt == 0:
            candidates = all_candidates
        else:
            # Shuffle within weight classes for diversity
            candidates = all_candidates[:]
            i = 0
            while i < len(candidates):
                j = i + 1
                while j < len(candidates) and candidates[j].bit_count() == candidates[i].bit_count():
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j
        
        words_list = [0]
        
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
            
            words_arr = np.array(words_list, dtype=np.uint32)
            xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
            bit_counts = popcount18(xor_vals)
            
            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)
        
        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]
        
        if best_len >= 500:
            break
    
    return best_words
