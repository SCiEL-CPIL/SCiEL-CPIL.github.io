import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Pre-compute popcount table for 17-bit integers
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    gf_poly = 0x1d  # x^8+x^4+x^3+x^2+1, primitive in GF(2^8)

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find element of order 17
    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    # Build generator polynomial g(x) of degree 8
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    # Generate all 512 words of the [17,9,5] code
    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    # Shorten: keep words with parity bit = 0
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)

    if best_size < 32:
        return [0]

    # Step 2: Targeted nonlinear extension using vectorized NumPy
    # Generate all 17-bit words at Hamming distance exactly 6 from zero
    # C(17,6) = 12376 candidates
    
    # Precompute bit positions for each combination
    bit_positions = list(range(17))
    
    # Generate all combinations of 6 positions out of 17
    candidate_pool = []
    for combo in combinations(bit_positions, 6):
        val = 0
        for pos in combo:
            val |= (1 << pos)
        candidate_pool.append(val)
    
    candidate_pool = np.array(candidate_pool, dtype=np.uint32)
    print(f"Step 2: Generated {len(candidate_pool)} candidates at distance 6 from zero")
    
    # Convert current code to numpy array for vectorized operations
    code_array = np.array(current_code, dtype=np.uint32)
    code_set = set(current_code)
    code_list = list(current_code)
    
    # For each candidate, check if it's compatible with all existing codewords
    # We need d(candidate, c) >= 6 for all c in code
    
    # Vectorized approach: for each candidate, compute XOR with all codewords
    # and check minimum popcount
    
    def vectorized_check_candidates(candidates, code_arr, existing_set):
        """
        Check which candidates are compatible with the current code.
        Returns list of valid candidates.
        """
        if len(candidates) == 0:
            return []
        
        valid_candidates = []
        
        # Process candidates in batches for memory efficiency
        batch_size = 1000
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            batch = candidates[start:end]
            
            # Skip candidates already in the code
            batch_mask = np.array([c not in existing_set for c in batch], dtype=bool)
            if not np.any(batch_mask):
                continue
            batch = batch[batch_mask]
            
            if len(batch) == 0:
                continue
            
            # Compute XOR: batch (m,1) XOR code (1,n) -> (m,n)
            # This gives us all pairwise XORs
            xor_matrix = batch[:, np.newaxis] ^ code_arr[np.newaxis, :]
            
            # Compute popcount for each element in xor_matrix
            # Use a lookup approach: convert to bytes and use precomputed table
            # For efficiency, use numpy's bit_count if available, otherwise manual
            
            # Manual popcount using bit manipulation
            # Process in chunks to avoid huge memory usage
            chunk_size = min(len(batch), 100)
            for chunk_start in range(0, len(batch), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch))
                chunk = xor_matrix[chunk_start:chunk_end]
                
                # Compute popcount for each row
                # For each row, find the minimum popcount
                # If min popcount >= 6, candidate is valid
                
                # Efficient popcount: use the fact that we only need to check if any
                # has popcount < 6
                # A word has popcount < 6 if it has at most 5 bits set
                
                # For each position in the chunk, check if any codeword is too close
                valid_in_chunk = []
                for i in range(len(chunk)):
                    row = chunk[i]
                    # Check if any element has popcount < 6
                    # Use bitwise trick: count bits
                    # For speed, we can early-exit
                    min_pc = 18  # max possible
                    for j in range(len(row)):
                        x = row[j]
                        # Count bits using lookup
                        pc = popcount[int(x) & 0x1FFFF]
                        if pc < min_pc:
                            min_pc = pc
                        if min_pc < 6:
                            break
                    if min_pc >= 6:
                        valid_in_chunk.append(batch[chunk_start + i])
                
                valid_candidates.extend(valid_in_chunk)
        
        return valid_candidates

    # Iteratively add valid candidates
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        
        # Get current code as numpy array
        code_arr = np.array(code_list, dtype=np.uint32)
        
        # Check all remaining candidates
        valid_new = vectorized_check_candidates(candidate_pool, code_arr, code_set)
        
        if not valid_new:
            break
        
        # Add all valid candidates
        for w in valid_new:
            code_set.add(w)
            code_list.append(w)
        
        added = len(valid_new)
        print(f"Round {round_num}: Added {added} candidates, total size: {len(code_list)}")
        
        if len(code_list) > best_size:
            best_size = len(code_list)
            best_code = list(code_list)
        
        # Remove added candidates from pool
        added_set = set(valid_new)
        candidate_pool = candidate_pool[~np.isin(candidate_pool, np.array(list(added_set), dtype=np.uint32))]
        
        if len(candidate_pool) == 0:
            break
    
    # Step 3: Local search with random candidates to find more
    # Use the same approach as before but with time budget
    
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 2.0:
        code_set = set(best_code)
        code_list = list(best_code)
        best_size = len(code_list)
        best_code_local = list(code_list)
        
        max_iters = 30000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            w = random.randint(0, 0x1FFFF)
            
            if w in code_set:
                continue
            
            ok = True
            for c in code_list:
                if popcount_fast(w ^ c) < 6:
                    ok = False
                    break
            
            if ok:
                code_set.add(w)
                code_list.append(w)
                if len(code_list) > best_size:
                    best_size = len(code_list)
                    best_code_local = list(code_list)
            else:
                # Try swap
                if len(code_list) > 50:
                    idx = random.randint(0, len(code_list) - 1)
                    u = code_list[idx]
                    code_set.remove(u)
                    code_list.pop(idx)
                    
                    ok2 = True
                    for c in code_list:
                        if popcount_fast(w ^ c) < 6:
                            ok2 = False
                            break
                    
                    if ok2 and w not in code_set:
                        code_set.add(w)
                        code_list.append(w)
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code_local = list(code_list)
                    else:
                        code_set.add(u)
                        code_list.insert(idx, u)
        
        if best_code_local:
            best_code = best_code_local
            best_size = len(best_code)
    
    # Final verification
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
            final = []
            for w in best_code:
                ok = True
                for v in final:
                    if popcount_fast(w ^ v) < 6:
                        ok = False
                        break
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
