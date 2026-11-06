import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the 256-word QR code seed
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    
    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = set()
    
    if beta:
        # Minimal polynomial of beta
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
            
        g_bits = []
        for c in poly_coeffs:
            g_bits.append(1 if c == 1 else 0)
            
        if len(g_bits) == 9 and g_bits[0] == 1:
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & ((1 << 17) - 1))
            
            shortened_codewords = list(set(shortened_codewords))
            
            # Verify
            min_d = 100
            for i in range(len(shortened_codewords)):
                for j in range(i+1, len(shortened_codewords)):
                    d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                    if d < min_d:
                        min_d = d
            
            if min_d >= 6:
                current_code = set(shortened_codewords)
            else:
                # Fallback if QR construction failed
                current_code = {0}
        else:
            current_code = {0}
    else:
        current_code = {0}

    if not current_code:
        current_code = {0}

    # 2. Vectorized Greedy Expansion
    code_list = list(current_code)
    
    def check_batch(candidates, code_list):
        """
        Returns list of indices in candidates that are valid.
        """
        if not code_list:
            return list(range(len(candidates)))
        
        C = np.array(code_list, dtype=np.uint32)
        K = np.array(candidates, dtype=np.uint32)
        
        XOR = K[:, None] ^ C[None, :]
        
        min_dists = np.full(XOR.shape[0], N + 1, dtype=np.int32)
        for b in range(N):
            bits = (XOR >> b) & 1
            dists = bits.sum(axis=1)
            min_dists = np.minimum(min_dists, dists)
            
        valid_mask = min_dists >= MIN_DIST
        return np.where(valid_mask)[0]

    BATCH_SIZE = 5000
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 15.0:
        candidates = rng.integers(0, 1 << N, size=BATCH_SIZE)
        
        code_set = set(code_list)
        unique_candidates = [int(c) for c in candidates if c not in code_set]
        
        if not unique_candidates:
            continue
            
        valid_indices = check_batch(unique_candidates, code_list)
        
        if len(valid_indices) > 0:
            for idx in valid_indices:
                new_word = unique_candidates[idx]
                current_code.add(new_word)
                code_list.append(new_word)
                
    # 3. Local Search / Swap Phase
    def is_valid_single(cand, code_list):
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True

    max_swaps = 2000
    swap_attempts = 0
    
    while time.time() < deadline - 2.0 and swap_attempts < max_swaps:
        if len(code_list) < 2:
            break
            
        remove_idx = rng.integers(0, len(code_list))
        w_removed = code_list[remove_idx]
        
        code_list.pop(remove_idx)
        current_code.remove(w_removed)
        
        added = 0
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            cand = int(rng.integers(0, 1 << N))
            if cand in current_code:
                continue
            if is_valid_single(cand, code_list):
                current_code.add(cand)
                code_list.append(cand)
                added += 1
                if added == 2:
                    break
        
        if added < 2:
            current_code.add(w_removed)
            code_list.append(w_removed)
            
        swap_attempts += 1

    return list(current_code)
