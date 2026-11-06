import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N

    # GF(256) arithmetic with polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
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

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = set()

    if beta:
        # Build minimal polynomial of beta over GF(2)
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

        # Check that coefficients are 0 or 1
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

        if valid_poly and len(g_bits) == 9 and g_bits[0] == 1:
            # Generate [17, 9] code
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)

            # Extend to 18
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)

            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & ((1 << 17) - 1))

            # Remove duplicates
            shortened_codewords = list(set(shortened_codewords))

            # Verify min distance
            min_d = 100
            for i in range(len(shortened_codewords)):
                for j in range(i + 1, len(shortened_codewords)):
                    d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                    if d < min_d:
                        min_d = d

            if min_d >= 6 and len(shortened_codewords) >= 256:
                current_code = set(shortened_codewords)

    # If we didn't get a good code from QR, fall back to greedy
    if len(current_code) < 10:
        code = set()
        code.add(0)
        random.seed(42)
        candidates = list(range(1, TOTAL))
        random.shuffle(candidates)
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            valid = True
            for w in code:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                code.add(cand)
        current_code = code

    # Convert to list for efficient access
    code_list = list(current_code)
    code_set = set(code_list)

    # Phase 1: Precompute conflicts using vectorized numpy
    # For each codeword, precompute which other words are too close
    # Use a bitset approach: for each word, store a bitmask of conflicting words
    
    # First, let's try a faster greedy addition using bit-parallel checks
    # We'll maintain a "forbidden" set of words that are too close to any codeword
    
    # Precompute a lookup: for each word, its neighbors within distance < 6
    # This is expensive, so we'll use a smarter approach
    
    # Phase 1: Greedy addition with random candidates, using early termination
    random.seed(123)
    added_count = 0
    max_add_attempts = 500000
    attempts = 0
    
    # Use a faster check: maintain a set of "blocked" words
    # A word is blocked if it's within distance < 6 of any codeword
    # We'll approximate this by checking against codewords directly but with early exit
    
    while time.time() < deadline - 15.0 and attempts < max_add_attempts:
        cand = random.randint(0, TOTAL - 1)
        if cand in code_set:
            attempts += 1
            continue
        
        # Fast conflict check with early termination
        valid = True
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            code_set.add(cand)
            code_list.append(cand)
            added_count += 1
        attempts += 1

    # Phase 2: Local search - try to replace codewords to free up space
    # Use a "swap" heuristic: remove one word, try to add two new words
    swap_improved = 0
    max_swap_rounds = 100000
    swap_attempts = 0
    best_code_size = len(code_list)
    
    while time.time() < deadline - 5.0 and swap_attempts < max_swap_rounds:
        if not code_list:
            break
        
        # Pick a random codeword to potentially remove
        idx = random.randint(0, len(code_list) - 1)
        old_word = code_list[idx]
        
        # Temporarily remove it
        code_set.discard(old_word)
        code_list.pop(idx)
        
        # Try to find new words that fit with the remaining code
        new_words = []
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            cand = random.randint(0, TOTAL - 1)
            if cand in code_set:
                continue
            valid = True
            for w in code_list:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                new_words.append(cand)
                code_set.add(cand)
                code_list.append(cand)
                if len(new_words) >= 2:
                    break
        
        # If we added more words than we removed, keep the improvement
        if len(new_words) > 1:
            swap_improved += 1
            if len(code_list) > best_code_size:
                best_code_size = len(code_list)
        else:
            # Put the old word back
            code_set.add(old_word)
            code_list.append(old_word)
            # Remove any new words we added
            for nw in new_words:
                code_set.discard(nw)
                if nw in code_list:
                    code_list.remove(nw)
        
        swap_attempts += 1

    # Phase 3: Another round of random additions
    attempts = 0
    while time.time() < deadline - 2.0 and attempts < 200000:
        cand = random.randint(0, TOTAL - 1)
        if cand in code_set:
            attempts += 1
            continue
        
        valid = True
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            code_set.add(cand)
            code_list.append(cand)
        attempts += 1

    return list(code_set)
