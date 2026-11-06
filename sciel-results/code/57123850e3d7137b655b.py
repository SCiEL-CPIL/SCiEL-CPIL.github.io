import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Construct the 256-word [17, 9, 6] code
    # This is the shortened [18, 9, 6] extended QR code.
    # We can generate it using the GF(256) method from the previous program, 
    # or simply hardcode a generator if we can find one.
    # Let's stick to the GF(256) generation as it is robust.
    
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
        # Fallback to seed
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
            
    # Remove duplicates
    code_set = set(shortened_codewords)
    if len(code_set) < 256:
        # Error in generation, fallback
        return list(code_set)
        
    # 2. Try to expand the code using Greedy Add
    # Convert to list for manipulation
    code_list = list(code_set)
    
    # Generate all possible 17-bit words
    # We only care about words NOT in the code
    # We can iterate through 0 to 2^17 - 1
    
    # To speed up, we can pre-compute the bit representation of all words? No, 131k is fine.
    
    # Optimization: 
    # For each candidate, we check distance against all current code words.
    # If we use NumPy, we can check a batch of candidates at once?
    # Or just loop.
    
    # Let's try a simple greedy loop first.
    # We will sort candidates by weight or just iterate 0..2^17-1.
    
    # To be more effective, we can try to add words that are "far" from the code.
    # But greedy add just checks compatibility.
    
    # We need to be careful with time.
    # 130,000 candidates * 256 checks = 33M distance checks.
    # Each check is a XOR and popcount.
    # In Python, this might take ~10-20 seconds.
    
    # Let's use NumPy to vectorize the distance check for a batch of candidates.
    
    # Create an array of all words not in the code
    all_words = np.arange(2**17, dtype=np.uint32)
    # Mask out existing code words
    # This is tricky with sets. 
    # We can create a boolean array of size 2^17
    mask = np.zeros(2**17, dtype=bool)
    for w in code_list:
        mask[w] = True
        
    candidates = all_words[~mask]
    
    # We will iterate through candidates.
    # To speed up, we can check if a candidate is compatible with the current code.
    # We can do this in chunks to use NumPy's broadcasting.
    
    # However, the code list changes as we add words.
    # So we can't pre-compute a single matrix.
    
    # Alternative: 
    # Maintain the code as a NumPy array.
    # For each candidate, compute distances to all code words.
    
    code_arr = np.array(code_list, dtype=np.uint32)
    
    added_count = 0
    # We can shuffle candidates to avoid bias? 
    # Or sort by weight? 
    # Let's just iterate.
    
    # To make it faster, we can check candidates in batches of, say, 1000.
    # But we need to stop if we add a word, because the code changes.
    # Actually, if we add a word, it might block other candidates.
    # So we must process one by one, or re-evaluate the batch.
    
    # Given the small gain (2 words), one-by-one is fine.
    
    # Optimization: 
    # Instead of checking against ALL code words, we can keep track of 
    # "conflict" bits? No, distance is not boolean.
    
    # Let's just loop.
    # We will break if time is running out.
    
    for w in candidates:
        if time.time() > deadline - 10:
            break
            
        # Check distance
        # XOR all code words with w
        xored = code_arr ^ w
        # Count bits
        # np.bit_count is available in newer NumPy? 
        # If not, use a lookup table or bin(x).count('1')
        # Let's use a fast popcount method.
        # For 17 bits, we can use a precomputed table for 8-bit chunks.
        
        # Actually, for a single candidate w, we can do:
        # dists = np.array([x.bit_count() for x in xored])
        # This is slow in Python.
        
        # Better:
        # Use a vectorized popcount.
        # We can split the 17-bit integers into 4 bytes (or 2 bytes + 1 byte).
        # Or just use `np.unpackbits`? That's for arrays.
        
        # Let's try a simple loop for the distance check, but optimized.
        # Since K is small (~256), a Python loop might be acceptable if we don't do it for 130k candidates.
        # 130k * 256 = 33M iterations. 
        # In Python, 33M iterations takes ~10-30s. 
        # We have 50s. It should fit.
        
        # But we can do better.
        # We can check if `min(dists) >= 6`.
        
        min_dist = 17
        # Iterate through code words
        # We can break early if we find a distance < 6
        for c in code_arr:
            d = (c ^ w).bit_count()
            if d < 6:
                min_dist = d
                break
        
        if min_dist >= 6:
            # Add w to code
            code_list.append(w)
            code_arr = np.append(code_arr, w)
            added_count += 1
            # Optional: check if we found 258
            if len(code_list) >= 258:
                # We can stop early if we are confident, but let's try to get more
                # Actually, 258 is the known lower bound. 
                # Can we get more? 
                # Let's continue searching for a bit.
                pass

    # If we didn't add any, or added some, we might try a local search.
    # But greedy add is the primary driver.
    
    return code_list
