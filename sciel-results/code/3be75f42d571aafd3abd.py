import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # 1. Construct the 256-word [17, 9, 6] code
    poly = 0x11D

    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()

    if beta is None:
        # Fallback to seed (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x)
    poly_coeffs = [1]
    current_beta = beta
    for _ in range(8):
        new_coeffs = [0] * (len(poly_coeffs) + 1)
        for i in range(len(poly_coeffs)):
            new_coeffs[i+1] = poly_coeffs[i]
        for i in range(len(poly_coeffs)):
            new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
        poly_coeffs = new_coeffs
        current_beta = gf_mul(current_beta, current_beta)

    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        g_bits[i] = 1 if c == 1 else 0

    # Generate 256 codewords
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)

    # Extend and shorten
    shortened_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))

    code_set = set(shortened_codewords)
    if len(code_set) < 256:
        return list(code_set)

    code_list = list(code_set)

    # 2. Greedy Add with Vectorized Distance Check
    # We will try to add words that are at distance exactly 5 from existing codewords.
    # This is a standard heuristic for extending codes.
    
    # Precompute the set of used words for O(1) lookup
    used = np.zeros(2**17, dtype=bool)
    for w in code_list:
        used[w] = True

    code_arr = np.array(code_list, dtype=np.uint32)
    
    # We'll maintain a list of "active" codewords (indices) to check against.
    # Initially all 256.
    active_indices = list(range(len(code_list)))
    
    # We want to find candidates.
    # Instead of iterating all 2^17, we can generate candidates from existing codewords.
    # A new word w must be at distance >= 6 from all existing.
    # This implies w is NOT in the radius-5 sphere of any existing word.
    
    # Let's try a different approach:
    # Iterate through all 2^17 words, but use vectorized distance check.
    # This allows us to check 1024 candidates at once against all 256+ codewords.
    
    # However, 131072 candidates * 256 codewords is 33M operations.
    # Vectorized: (1024 x 256) matrix of XORs, then bit_count.
    # 131072 / 1024 = 128 iterations.
    # Each iteration: 1024*256 XORs + bit_counts + min.
    # This is very fast in NumPy.
    
    candidates = np.arange(2**17, dtype=np.uint32)
    candidates = candidates[~used]
    
    # Sort candidates by weight? Maybe lower weight is better? 
    # Or just random? 
    # Let's try sorting by weight to see if it helps.
    # Weights are 0..17.
    # Actually, let's just process in chunks.
    
    chunk_size = 1024
    n_candidates = len(candidates)
    
    for i in range(0, n_candidates, chunk_size):
        if time.time() > deadline - 5:
            break
            
        chunk = candidates[i:i+chunk_size]
        
        # Compute distances from chunk to all active codewords
        # shape: (chunk_size, num_active)
        # We need min distance to be >= 6
        
        # To save memory and time, we can check against code_arr directly.
        # code_arr grows, but we can just use the current list.
        
        # Let's use a loop over the chunk to find compatible words?
        # No, let's vectorize the check against the CURRENT code_arr.
        
        # Expand chunk to (chunk_size, 1) and code_arr to (1, n_code)
        # XOR: (chunk_size, n_code)
        # Bit count: (chunk_size, n_code)
        # Min along axis 1: (chunk_size,)
        
        # This creates a large temporary array. 
        # 1024 * 256 = 262,144 uint32s = 1MB. Fine.
        # As code grows, it gets bigger.
        # If code reaches 300, 1024*300 = 307,200. Still fine.
        
        # Let's try it.
        
        # We need to be careful with uint32 overflow? No, XOR is bitwise.
        
        # Note: np.bitwise_xor on uint32 works.
        # np.bit_count is not directly available in older NumPy? 
        # Actually, np.bit_count exists in NumPy 1.24+. 
        # If not, we can use a lookup table or a different method.
        # Let's assume modern NumPy.
        
        # Alternative: Use a precomputed lookup table for byte popcounts?
        # That might be faster if np.bit_count is slow.
        # But let's try np.bit_count first.
        
        try:
            # Vectorized distance check
            # We need to check if ANY codeword is within distance 5.
            # i.e., min_distance >= 6.
            
            # To avoid huge memory if code gets large, we might need to chunk code_arr too?
            # 256 is small. Let's assume code stays under 1000.
            # 1024 * 1000 = 1M uint32s = 4MB. Acceptable.
            
            # Let's do it in smaller chunks of code if necessary?
            # No, let's just do it.
            
            # We'll check the chunk against the current code_arr.
            # If a candidate is compatible, we add it.
            # But adding it changes code_arr for the next candidate in the same chunk?
            # Yes. So we should process candidates one by one if we want to be strict?
            # Or we can find ALL compatible candidates in the chunk against the CURRENT code,
            # add them all, and then re-verify?
            # Greedy: Add one, update code, check next.
            # This is slower but safer.
            
            # However, if we check the whole chunk at once, we might add two words that are too close to each other.
            # So we must check pairwise distance among newly added words too.
            
            # Strategy:
            # 1. Find all candidates in chunk that are dist >= 6 from current code.
            # 2. From that set, greedily add words that are dist >= 6 from each other.
            
            # Step 1: Vectorized check against current code.
            # We can do this efficiently.
            
            # Let's just do the simple loop for now, it might be fast enough with PyPy/NumPy overhead?
            # No, Python loop is slow.
            
            # Let's use the vectorized check to FILTER candidates, then do a small greedy loop on the survivors.
            
            # Filter:
            # dists = np.bit_count(chunk[:, None] ^ code_arr[None, :])
            # min_dists = np.min(dists, axis=1)
            # valid_indices = np.where(min_dists >= 6)[0]
            
            # This is fast.
            
            # But we need to handle the case where code_arr is empty? No, it's 256.
            
            # Let's try this.
            
            # To handle growing code_arr, we might need to update it.
            # If we add a word, code_arr changes.
            # So we should probably just iterate through the chunk, and for each candidate,
            # check against the CURRENT code_arr (which includes newly added words).
            # But that defeats the purpose of vectorization.
            
            # Compromise:
            # Process the chunk.
            # For each candidate in the chunk (in order):
            #   Check against current code_arr (vectorized single row vs column? No, that's scalar).
            #   Actually, checking one candidate against 256 codewords is fast even in Python?
            #   256 XORs + bit counts. ~50us?
            #   131k candidates * 50us = 6.5s.
            #   This is actually acceptable!
            #   The previous code was doing this but with Python loops for the inner check.
            #   Let's optimize the inner check using NumPy for a single candidate.
            
            # Let's revert to the loop but use NumPy for the distance check of a single candidate.
            
            pass # Placeholder for logic below
            
        except Exception:
            pass

    # Let's implement the "Single candidate vectorized check" approach.
    # It's simpler and likely fast enough.
    
    # Reset for this strategy
    # We already have candidates and used.
    
    for w in candidates:
        if time.time() > deadline - 5:
            break
            
        # Check w against all codewords in code_arr
        # w is a Python int (or np.uint32 scalar)
        # code_arr is np.uint32 array
        
        # Convert w to uint32 for numpy
        w_np = np.uint32(w)
        
        # Compute XORs
        xor_vals = code_arr ^ w_np
        # Compute bit counts
        # np.bit_count is available in NumPy 1.24+
        # If not, we can use a trick or fall back to Python.
        # Let's assume it's there.
        
        try:
            dists = np.bit_count(xor_vals)
        except AttributeError:
            # Fallback for older NumPy
            # This will be slow, but we assume modern env.
            dists = np.array([x.bit_count() for x in xor_vals], dtype=np.uint8)
            
        # Check if all distances >= 6
        if np.min(dists) >= 6:
            code_list.append(w)
            code_arr = np.append(code_arr, w)
            used[w] = True

    return code_list
