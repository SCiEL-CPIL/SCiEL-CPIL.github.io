import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N
    MAX_WORDS = TOTAL  # Upper bound

    # Precompute popcount for 17-bit numbers? 
    # Actually, numpy bit_count is fast enough for vectors.
    # But we can precompute a lookup table for 16-bit and add 1 bit, or just use np.bitwise_count if available (Python 3.10+)
    # Since we don't know the Python version, we use a trick: 
    # popcount(x) = popcount(x & 0xFFFF) + popcount(x >> 16)
    # We can precompute popcount for 0..65535.
    
    popcount_16 = np.zeros(1 << 16, dtype=np.int32)
    for i in range(1, 1 << 16):
        popcount_16[i] = popcount_16[i >> 1] + (i & 1)
    
    def vectorized_popcount(x):
        # x is a numpy array of uint32 or int32
        # x is in range [0, 2**17)
        low = x & 0xFFFF
        high = (x >> 16) & 0x1
        return popcount_16[low] + high

    # Helper to generate the QR code as a seed
    def get_qr_code():
        # GF(256) arithmetic
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 0x100:
                    a ^= 0x11D
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

        beta = None
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                beta = candidate
                break
        
        if beta is None:
            return set()

        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i + 1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)

        g_bits = []
        valid_poly = True
        for c in poly_coeffs:
            if c == 0:
                g_bits.append(0)
            elif c == 1:
                g_bits.append(1)
            else:
                valid_poly = False
                break

        if not (valid_poly and len(g_bits) == 9 and g_bits[0] == 1):
            return set()

        codewords = set()
        for u in range(512):
            c = 0
            for k in range(9):
                if (u >> k) & 1:
                    for j in range(9):
                        if g_bits[j]:
                            c ^= 1 << (j + k)
            # Shorten: we need 17 bits. The generator is degree 9, so codewords are 17 bits?
            # The [18,9,6] code is extended. Shortening one coordinate gives [17,8,6] or [17,9,6]?
            # The task says "256-word linear reference is the [18,9,6] ... shortened by one coordinate".
            # [18,9,6] has 512 words. Shortening fixes one bit to 0. If we fix the parity bit to 0, we get a [17,8,6] code of size 256?
            # Wait, if we extend [17,9,6] to [18,9,6], the extension bit is parity.
            # If we shorten [18,9,6] by fixing the last bit to 0, we get all codewords of [17,9,6] with even parity?
            # No, the [17,9,6] code is not necessarily all even parity.
            # Let's just generate the 512 codewords of the [17,9,6] code directly if possible, or use the 256 words from shortening.
            # The prompt says "256-word linear reference". So we aim for 256.
            # Let's verify the distance of the generated 17-bit words.
            codewords.add(c)
        
        # Verify min distance
        min_d = 100
        cw_list = list(codewords)
        for i in range(len(cw_list)):
            for j in range(i+1, len(cw_list)):
                d = (cw_list[i] ^ cw_list[j]).bit_count()
                if d < min_d:
                    min_d = d
        if min_d >= 6:
            return codewords
        return set()

    # Initialize code
    seed_code = get_qr_code()
    if len(seed_code) < 10:
        # Fallback to random seed if QR fails
        random.seed(42)
        seed_code = {random.randint(0, TOTAL - 1) for _ in range(10)}
        # Ensure distinct
        while len(seed_code) < 10:
            seed_code.add(random.randint(0, TOTAL - 1))
            
    current_code = list(seed_code)
    current_set = set(current_code)
    
    # Precompute forbidden zones for the seed? 
    # We will maintain a list of forbidden masks.
    # forbidden_mask[i] is a set of words that are too close to current_code[i].
    # But storing sets is slow. 
    # Instead, for each candidate, we check distance to all current codewords.
    # Vectorized check:
    #   cand_arr: shape (B, 1)
    #   code_arr: shape (K, 1)
    #   diff = cand_arr ^ code_arr
    #   dists = vectorized_popcount(diff)
    #   valid = np.all(dists >= MIN_DIST, axis=1)
    
    code_arr = np.array(current_code, dtype=np.uint32)
    
    batch_size = 1024
    random.seed(123)
    
    # Main loop: Greedy addition
    # We try to add words. If we can't add, we try swaps.
    
    time_limit_add = deadline - 15.0
    while time.time() < time_limit_add:
        # Generate batch of candidates
        candidates = np.array([random.randint(0, TOTAL - 1) for _ in range(batch_size)], dtype=np.uint32)
        
        # Filter out candidates already in code (optional optimization, but set lookup is O(1) per item, vectorized is tricky)
        # We can just check distances. If a candidate is in code, distance is 0, so it will be rejected.
        
        # Vectorized distance check
        # To avoid creating a huge (B, K) array, we can process in chunks if K is large.
        # K starts at 256. B=1024. 1024*256 = 262k elements. Very fast.
        
        # Reshape for broadcasting
        # cands: (B, 1)
        # code: (K, 1)
        # We want (B, K) distances.
        
        # We can do this in chunks of code if K is very large, but K is small.
        dists = vectorized_popcount(candidates[:, np.newaxis] ^ code_arr[np.newaxis, :])
        
        # valid_mask[i] is True if candidate i is at distance >= MIN_DIST from all codewords
        valid_mask = np.all(dists >= MIN_DIST, axis=1)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Heuristic: Pick the candidate that maximizes the minimum distance to existing codewords?
            # This is "Maximum Residual Value".
            # min_dists[i] = min(dists[i, :])
            # We want to maximize min_dists[i] for valid i.
            # This tends to pack words more evenly.
            
            min_dists = np.min(dists[valid_mask], axis=1)
            # Pick the best one
            best_idx_in_valid = np.argmax(min_dists)
            best_cand_idx = valid_indices[best_idx_in_valid]
            best_cand = candidates[best_cand_idx]
            
            current_code.append(best_cand)
            current_set.add(best_cand)
            code_arr = np.append(code_arr, best_cand)
            
            # If we found multiple valid candidates, we could add more?
            # Usually, adding one changes the landscape. 
            # But we can add all valid ones if they are far from each other too?
            # That requires checking mutual distances. 
            # For now, let's just add the best one and restart the loop.
            # This is conservative but safe.
            
    # Phase 2: Swap-based local search
    # Try to replace a word with a new one that allows adding more words later.
    # Or simply: try to find a word that is valid for the code minus one word, and is "better" than the removed word.
    # "Better" is hard to define. 
    # Alternative: Randomly remove a word, try to add a new word. If we can add a new word, keep it. 
    # This effectively allows the code to "move" around the space.
    
    time_limit_swap = deadline - 5.0
    swap_batch_size = 512
    
    while time.time() < time_limit_swap:
        if len(current_code) == 0:
            break
            
        # Pick a random word to remove
        idx_to_remove = random.randint(0, len(current_code) - 1)
        old_word = current_code[idx_to_remove]
        
        # Remove it
        current_code.pop(idx_to_remove)
        current_set.discard(old_word)
        code_arr = np.array(current_code, dtype=np.uint32)
        
        # Try to find a new word
        candidates = np.array([random.randint(0, TOTAL - 1) for _ in range(swap_batch_size)], dtype=np.uint32)
        
        if len(code_arr) > 0:
            dists = vectorized_popcount(candidates[:, np.newaxis] ^ code_arr[np.newaxis, :])
            valid_mask = np.all(dists >= MIN_DIST, axis=1)
        else:
            valid_mask = np.ones(len(candidates), dtype=bool)
            
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Pick the best one (max min distance)
            if len(code_arr) > 0:
                min_dists = np.min(dists[valid_mask], axis=1)
                best_idx_in_valid = np.argmax(min_dists)
            else:
                best_idx_in_valid = 0
            best_cand_idx = valid_indices[best_idx_in_valid]
            new_word = candidates[best_cand_idx]
            
            # Add the new word
            current_code.append(new_word)
            current_set.add(new_word)
            code_arr = np.append(code_arr, new_word)
        else:
            # Put the old word back
            current_code.append(old_word)
            current_set.add(old_word)
            code_arr = np.append(code_arr, old_word)
            
    # Final check and return
    # Ensure all are distinct and valid (sanity check, though logic should guarantee it)
    # The set ensures distinctness.
    # Distances are checked during insertion.
    
    return list(current_set)

if __name__ == "__main__":
    # Quick self-test
    code = construct()
    print(f"Found {len(code)} codewords")
    # Verify
    min_d = 100
    for i in range(len(code)):
        for j in range(i+1, len(code)):
            d = (code[i] ^ code[j]).bit_count()
            if d < min_d:
                min_d = d
    print(f"Min distance: {min_d}")
