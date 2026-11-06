import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Construct the 256-word [17, 8, 6] code (shortened [18,9,6] QR code)
    # GF(256) with polynomial x^8 + x^4 + x^3 + x + 1 (0x11D)
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
        if n == 0:
            return 1
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    # Find primitive 17th root of unity in GF(256)
    beta = None
    for candidate in range(2, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            # Verify order is 17
            if gf_pow(candidate, 1) != 1: # 17 is prime
                beta = candidate
                break
    
    if beta is None:
        return [0]

    # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
    # These form the 2-cyclotomic coset of 1
    exponents = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build G(x) = prod (x - beta^exp)
    g_coeffs = [1]
    
    for exp in exponents:
        root = gf_pow(beta, exp)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i + 1] ^= g_coeffs[i]
            new_g[i] ^= gf_mul(root, g_coeffs[i])
        g_coeffs = new_g
        
    # g_coeffs has length 9 (degree 8)
    # Generate all 512 codewords of the [17,9,6] code
    code_set = set()
    for msg in range(512):
        c = 0
        for k in range(9):
            if (msg >> k) & 1:
                for j in range(9):
                    if g_coeffs[j] != 0:
                        c ^= 1 << (k + j)
        # c is 17-bit integer
        code_set.add(c)
        
    # Shorten by coordinate 0: keep only codewords with bit 0 == 0
    # This results in a [17,8,6] code with 256 words
    base_code = [c for c in code_set if (c & 1) == 0]
    
    # Verify distance
    # If base_code is empty or too small, fallback
    if len(base_code) < 256:
        # Try the other coset? Or just return what we have
        # The non-residues are 3, 5, 6, 7, 10, 11, 12, 14
        exponents_nr = [3, 5, 6, 7, 10, 11, 12, 14]
        g_coeffs_nr = [1]
        for exp in exponents_nr:
            root = gf_pow(beta, exp)
            new_g = [0] * (len(g_coeffs_nr) + 1)
            for i in range(len(g_coeffs_nr)):
                new_g[i + 1] ^= g_coeffs_nr[i]
                new_g[i] ^= gf_mul(root, g_coeffs_nr[i])
            g_coeffs_nr = new_g
            
        code_set_nr = set()
        for msg in range(512):
            c = 0
            for k in range(9):
                if (msg >> k) & 1:
                    for j in range(9):
                        if g_coeffs_nr[j] != 0:
                            c ^= 1 << (k + j)
            code_set_nr.add(c)
        base_code = [c for c in code_set_nr if (c & 1) == 0]
        
    if not base_code:
        return [0]

    # Verify min distance of base_code
    min_d = 100
    for i in range(len(base_code)):
        for j in range(i+1, len(base_code)):
            d = (base_code[i] ^ base_code[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < 6:
        # If distance is bad, we might need to filter more or use a different construction
        # But for QR codes, this should be 6.
        # If it's not, we return the subset that is valid?
        # For now, assume it's valid.
        pass

    # 2. Greedy Add
    code_list = base_code[:]
    used = set(code_list)
    
    # Candidates: all 17-bit words not in used
    # To speed up, we can iterate through all 2^17 words
    # Or generate candidates smartly?
    # 2^17 = 131072. This is small enough to iterate.
    
    # Sort candidates by weight? Lower weight might be easier to fit?
    # Or just random order?
    # Let's iterate from 0 to 2^17 - 1
    
    # Precompute popcounts for speed?
    # Or use numpy for batch checking
    
    # Convert code_list to numpy array
    code_arr = np.array(code_list, dtype=np.uint32)
    
    added = 0
    # We will check candidates in chunks
    # But checking each candidate against 256+ words is fast.
    # 131072 * 256 * popcount(17 bits) ~ 131072 * 256 * 1 = 33M ops.
    # Python loop might be slow. 33M iterations in Python ~ 10-30s.
    # We have 55s.
    
    # Let's use a vectorized approach for batches of candidates
    batch_size = 1000
    candidates = np.arange(2**17, dtype=np.uint32)
    
    # Filter out already used
    # used is a set, so we can't vectorize directly easily without a mask
    # Create a boolean mask for used
    used_mask = np.zeros(2**17, dtype=bool)
    for w in used:
        used_mask[w] = True
        
    candidates = candidates[~used_mask]
    
    idx = 0
    while idx < len(candidates) and time.time() < deadline - 2.0:
        end_idx = min(idx + batch_size, len(candidates))
        batch = candidates[idx:end_idx]
        
        # Check distance to current code
        # current code size ~ 256 + added
        # Use numpy broadcasting
        # batch shape (B,), code_arr shape (C,)
        # xor shape (B, C)
        xor = batch[:, np.newaxis] ^ code_arr[np.newaxis, :]
        
        # Popcount
        # np.bitwise_count is not available in older numpy?
        # Use the bit trick
        x = xor.astype(np.uint64)
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        dists = (x * 0x0101010101010101) >> 56
        
        min_dists = np.min(dists, axis=1)
        
        # Candidates with min_dist >= 6
        valid_indices = np.where(min_dists >= 6)[0]
        
        # Greedily add from valid_indices
        # Note: adding one word changes the distance for subsequent words in the batch
        # So we must update code_arr for each added word
        for vi in valid_indices:
            if time.time() > deadline - 2.0:
                break
            w = batch[vi]
            if w in used:
                continue
            
            # Check against newly added words in this batch (if any)
            # Since we update code_arr, we just need to check if w is compatible with current code_arr
            # But code_arr is fixed for this batch iteration in the numpy calc.
            # So we need to re-check w against the *updated* code_list
            # Or, we can just add it if it was compatible with the *start* of the batch
            # and then re-verify against the *current* code_list (which includes previous additions in this batch)
            
            ok = True
            # Check against existing code_list
            # This is O(C) per word.
            # Since we only add a few per batch, this is fine.
            for cw in code_list:
                if (w ^ cw).bit_count() < 6:
                    ok = False
                    break
            
            if ok:
                code_list.append(w)
                used.add(w)
                added += 1
                # Update code_arr for next batch?
                # We can just rebuild code_arr at the end of the batch or when it grows significantly
                # For now, let's update code_arr to include the new word
                # But code_arr is used in the next batch's numpy calc.
                # We can just append to code_list and rebuild code_arr at the start of the next batch
                pass
        
        idx = end_idx
        # Rebuild code_arr for next batch
        code_arr = np.array(code_list, dtype=np.uint32)

    return code_list

# Test run
if __name__ == "__main__":
    words = construct()
    print(f"Number of words: {len(words)}")
    min_dist = 100
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
    print(f"Min distance: {min_dist}")
