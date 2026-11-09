import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # --- Attempt 1: Construct 512-word code from QR(23) ---
    # QR(23) is [23, 12, 7].
    # Generator polynomial g(x) = x^11 + x^9 + x^7 + x^6 + x^5 + x + 1
    g = 0
    for bit in (0, 1, 5, 6, 7, 9, 11):
        g |= 1 << bit
        
    # Generate all 4096 codewords
    # c(x) = m(x) * g(x)
    # m(x) has degree < 12
    qr_words = []
    for m in range(4096):
        c = 0
        mm = m
        while mm:
            if mm & 1:
                c ^= g
            mm >>= 1
            g <<= 1
        # c is now the product, degree < 23
        qr_words.append(c)
        
    # Shorten by 3 positions (0, 1, 2)
    # Keep words with bits 0, 1, 2 == 0
    mask_shorten = 0b111
    shortened = []
    for w in qr_words:
        if (w & mask_shorten) == 0:
            # Remove bits 0, 1, 2
            new_w = (w >> 3) & ((1 << 20) - 1)
            shortened.append(new_w)
            
    # shortened should have 512 words of length 20
    if len(shortened) != 512:
        # Fallback if construction failed unexpectedly
        pass
    else:
        # Puncture 2 positions (0, 1) from the 20-bit words
        # Remove bits 0, 1
        punctured = []
        for w in shortened:
            # Shift right by 2 to remove bits 0,1
            new_w = (w >> 2) & ((1 << 18) - 1)
            punctured.append(new_w)
            
        # Check for duplicates (should be none if code is linear and puncturing is injective on the subcode)
        # Actually, puncturing a linear code can cause collisions if the code has codewords with support only in punctured positions.
        # But min distance 7 means no codeword has weight < 7.
        # Punctured positions are 2. No codeword of weight 7 can be supported entirely in 2 positions.
        # So the map is injective.
        
        # Verify min distance
        # 512 words. Pairwise check is 512*511/2 ~ 130k checks.
        # Fast enough.
        valid = True
        words_list = punctured
        for i in range(len(words_list)):
            if time.time() > deadline - 5.0:
                valid = False
                break
            for j in range(i + 1, len(words_list)):
                d = (words_list[i] ^ words_list[j]).bit_count()
                if d < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            return words_list

    # --- Attempt 2: Fallback to Greedy Search ---
    # Use the 64-word Golay-shortened code as seed
    # Or just use the greedy search from Inspiration 1
    
    # Fast popcount for 18-bit values using 3x6-bit lookup
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
    
    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)
        
    # Sort candidates by weight
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))
    
    best_words = [0]
    best_len = 1
    
    rng = np.random.default_rng(42)
    
    # Try a few restarts
    for attempt in range(10):
        if time.time() > deadline - 5.0:
            break
            
        if attempt == 0:
            candidates = all_candidates
        else:
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
