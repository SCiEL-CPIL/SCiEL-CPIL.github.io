import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Step 1: Construct the 256-word [17, 8, 6] linear code
    # Derived from the [18, 9, 6] extended quadratic-residue code shortened by one coordinate
    
    poly = 0x1D  # x^8 + x^4 + x^3 + x^2 + 1
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
        
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
        
    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        return [0]
        
    # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial by multiplying (x - alpha^q) for each quadratic residue q
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # g_coeffs should have length 9 (degree 8), coefficients in GF(2)
    # Build generator matrix G (9 x 17)
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if j < len(g_coeffs) and g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
                
    # Generate all 512 codewords of the [17, 9, 5] code
    # Then extend to [18, 9, 6] by adding parity bit
    # Then shorten by removing words with parity bit = 1, giving [17, 8, 6]
    words_short = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        # Compute parity
        parity = int(np.sum(word) % 2)
        # If parity is 0, this word is in the shortened code
        if parity == 0:
            val = 0
            for i in range(17):
                if word[i]:
                    val |= (1 << i)
            words_short.append(val)
            
    if not words_short:
        return [0]
        
    current_code = np.array(words_short, dtype=np.int64)
    
    # Step 2: Precompute popcount table for fast distance checks
    # This enables O(1) Hamming distance lookups
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    
    best_code = current_code.copy()
    best_size = len(current_code)
    
    # Step 3: Greedy extension with multiple restarts
    # Try to add words from the remaining 2^17 - 256 words
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
            
        # Create a shuffled list of candidates
        rng = np.random.default_rng(seed=restart * 12345 + 67890)
        candidates = rng.permutation(2**17)
        
        current_list = list(current_code)
        current_set = set(current_list)
        
        # Precompute current codewords as numpy array for fast XOR
        cur_arr = np.array(current_list, dtype=np.int64)
        
        # Greedy: try to add each candidate if it's at distance >= 6 from all current words
        for w in candidates:
            if time.time() > deadline - 3.0:
                break
                
            w_int = int(w)
            if w_int in current_set:
                continue
                
            # Vectorized: compute XOR of w with all codewords, then lookup popcount
            diffs = cur_arr ^ w_int
            dists = popcount_table[diffs]
            min_d = int(np.min(dists))
            
            if min_d >= 6:
                current_list.append(w_int)
                current_set.add(w_int)
                # Update array in-place by appending
                cur_arr = np.append(cur_arr, w_int)
                
        # Local improvement: try to swap out a codeword for a better one
        # This can help escape local optima
        if len(current_list) > 0 and time.time() < deadline - 3.0:
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                cur_arr = np.array(current_list, dtype=np.int64)
                for i in range(len(current_list)):
                    if time.time() > deadline - 3.0:
                        break
                    # Try removing codeword i and see if we can add a better replacement
                    remaining = current_list[:i] + current_list[i+1:]
                    rem_arr = np.array(remaining, dtype=np.int64)
                    # Find a word that's far from all remaining
                    # Use a small random sample of candidates
                    sample_size = min(500, 2**17 - len(remaining))
                    sample = rng.choice(2**17, size=sample_size, replace=False)
                    for w_int in sample:
                        w_int = int(w_int)
                        if w_int in set(remaining):
                            continue
                        diffs = rem_arr ^ w_int
                        dists = popcount_table[diffs]
                        if int(np.min(dists)) >= 6:
                            # Check if this word is "better" than the removed one
                            # (e.g., has more distance to others on average)
                            new_list = remaining + [w_int]
                            new_set = set(new_list)
                            if len(new_set) == len(remaining) + 1:
                                # Replace
                                current_list = new_list
                                current_set = new_set
                                improved = True
                                break
                    if improved:
                        break
                        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_code = np.array(current_list, dtype=np.int64)
            
    return list(best_code)
