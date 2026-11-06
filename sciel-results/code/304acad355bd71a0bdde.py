import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    MASK = (1 << N) - 1
    
    # Precompute bit counts for all 17-bit values (2^17 = 131072 entries)
    bit_count_table = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        bit_count_table[i] = bit_count_table[i >> 1] + (i & 1)
    
    def popcount(x):
        """Fast popcount using precomputed table."""
        return bit_count_table[x]
    
    def hamming_dist(a, b):
        return popcount(a ^ b)
    
    # GF(256) construction for the QR code
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

    def build_qr_code():
        """Build the [17,8,6] shortened extended QR code."""
        # Find primitive 17th root of unity in GF(256)
        beta = None
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                beta = candidate
                break
        
        if beta is None:
            return None
        
        # Build generator polynomial g(x) = prod_{i=1}^{8} (x - beta^{2^i})
        # The roots are beta^1, beta^2, beta^4, beta^8, beta^16, beta^32, beta^64, beta^128
        # Since 255 = 15*17, the conjugacy class of beta has size 8
        
        poly_coeffs = [1]  # Start with 1
        current_root = beta
        for _ in range(8):
            # Multiply by (x - current_root)
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] = poly_coeffs[i]
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_root)
            poly_coeffs = new_coeffs
            current_root = gf_mul(current_root, current_root)
        
        # Extract bits from GF(256) coefficients
        g_bits = []
        for c in poly_coeffs:
            # Convert GF(256) element to 8-bit representation
            g_bits.extend((c >> i) & 1 for i in range(8))
        
        # The generator polynomial should have degree 8 (9 coefficients)
        # But we need to extract the binary representation properly
        # For a [17,8,6] code, g(x) has degree 9, so 10 bits
        
        # Actually, let's use a different approach: directly enumerate
        # the [17,8,6] code using the parity-check matrix
        
        # The [18,9,6] extended QR code has parity-check matrix H of size 9x18
        # We'll construct it from the roots
        
        # Alternative: use the known structure
        # The [17,8,6] code is the shortening of [18,9,6]
        
        # Let's try a direct construction using the binary representation
        # of the generator polynomial coefficients
        
        # For GF(256) with primitive polynomial x^8+x^4+x^3+x^2+1:
        # The generator polynomial of the [17,8,6] QR code
        # has coefficients that are GF(256) elements
        
        # Let me use a simpler approach: generate all 2^8 = 256 codewords
        # by systematic encoding using the generator polynomial
        
        # The generator polynomial in binary form (9 bits for degree 8)
        # For the QR code, g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
        # Let's verify this is correct by checking if it's divisible by the minimal polynomial
        
        # Actually, let's just use a known valid [17,8,6] code
        # The binary QR code of length 17 has generator polynomial:
        # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
        
        g_poly_bits = [1, 1, 0, 1, 1, 1, 0, 1, 1]  # MSB first: x^8+x^7+x^5+x^4+x^3+x+1
        # Verify: this should give a [17,8,6] code
        
        # Generate all 256 codewords
        codewords = []
        for u in range(256):
            c = 0
            for k in range(8):
                if (u >> k) & 1:
                    for j in range(9):
                        if g_poly_bits[j]:
                            c ^= 1 << (j + k)
            codewords.append(c & MASK)
        
        # Remove duplicates and verify
        codewords = list(set(codewords))
        
        # Verify minimum distance
        if len(codewords) != 256:
            return None
        
        min_d = 100
        for i in range(len(codewords)):
            for j in range(i+1, len(codewords)):
                d = hamming_dist(codewords[i], codewords[j])
                if d < min_d:
                    min_d = d
                if min_d < 6:
                    return None
        
        if min_d >= 6:
            return set(codewords)
        
        return None
    
    # Try to build QR code
    current_code = build_qr_code()
    
    if current_code is None:
        # Fallback: greedy construction
        current_code = set()
        current_code.add(0)
        
        random.seed(42)
        candidates = list(range(1, 1 << N))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline - 20.0:
                break
            valid = True
            for w in current_code:
                if hamming_dist(cand, w) < MIN_DIST:
                    valid = False
                    break
            if valid:
                current_code.add(cand)
    
    # Local search to improve the code
    random.seed(42)
    code_list = list(current_code)
    code_set = set(code_list)
    
    # Strategy 1: Try to add new codewords by mutating existing ones
    # For each existing codeword, flip 1-3 bits and check if the result is valid
    
    def try_add_candidate(cand, code_set):
        """Try to add a candidate to the code set. Return True if added."""
        if cand in code_set:
            return False
        for w in code_set:
            if hamming_dist(cand, w) < MIN_DIST:
                return False
        code_set.add(cand)
        return True
    
    # Generate candidates by mutating existing codewords
    added_count = 0
    max_add_rounds = 100
    
    for round_num in range(max_add_rounds):
        if time.time() > deadline - 15.0:
            break
        
        improved_this_round = False
        code_list = list(code_set)
        random.shuffle(code_list)
        
        for word in code_list:
            if time.time() > deadline - 15.0:
                break
            
            # Try flipping 1, 2, or 3 bits
            for n_flips in [1, 2, 3]:
                # Generate random flip patterns
                for _ in range(20):
                    if time.time() > deadline - 15.0:
                        break
                    
                    flip_mask = 0
                    for _ in range(n_flips):
                        bit_pos = random.randint(0, N-1)
                        flip_mask ^= (1 << bit_pos)
                    
                    cand = word ^ flip_mask
                    if cand in code_set:
                        continue
                    
                    # Check if cand is valid
                    valid = True
                    for w in code_set:
                        if hamming_dist(cand, w) < MIN_DIST:
                            valid = False
                            break
                    
                    if valid:
                        code_set.add(cand)
                        added_count += 1
                        improved_this_round = True
                        break  # Move to next word
        
        if not improved_this_round:
            # No improvements this round, try a different strategy
            # Random sampling
            for _ in range(100):
                if time.time() > deadline - 15.0:
                    break
                cand = random.randint(0, MASK)
                if cand not in code_set:
                    valid = True
                    for w in code_set:
                        if hamming_dist(cand, w) < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        code_set.add(cand)
                        added_count += 1
    
    # Strategy 2: Swap-based local search
    # Try to swap out poorly-placed codewords for better ones
    swap_rounds = 30
    for round_num in range(swap_rounds):
        if time.time() > deadline - 10.0:
            break
        
        code_list = list(code_set)
        random.shuffle(code_list)
        improved = False
        
        for old_word in code_list:
            if time.time() > deadline - 10.0:
                break
            
            # Try a few swap candidates generated by mutating other codewords
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                
                # Pick a random other codeword and mutate it
                other_word = random.choice(code_list)
                if other_word == old_word:
                    continue
                
                n_flips = random.choice([1, 2, 3])
                flip_mask = 0
                for _ in range(n_flips):
                    bit_pos = random.randint(0, N-1)
                    flip_mask ^= (1 << bit_pos)
                
                new_word = other_word ^ flip_mask
                if new_word == old_word or new_word in code_set:
                    continue
                
                # Check if new_word is valid against all other codewords (excluding old_word)
                valid = True
                for w in code_set:
                    if w == old_word:
                        continue
                    if hamming_dist(new_word, w) < MIN_DIST:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(old_word)
                    code_set.add(new_word)
                    improved = True
                    break
        
        if not improved:
            break
    
    return list(code_set)
