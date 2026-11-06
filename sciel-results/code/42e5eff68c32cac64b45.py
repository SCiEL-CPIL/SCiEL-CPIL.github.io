import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    N = 17
    MIN_DIST = 6
    MASK = (1 << N) - 1

    # 1. Construct the [17,9,6] shortened extended QR code (256 words)
    # This is the linear reference code.
    
    # GF(256) operations
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
        # 255 = 15 * 17. We need an element of order 17.
        # alpha^k where 255/k is a multiple of 17.
        # Let's just scan for an element of order 17.
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    
    code_set = set()
    
    if beta:
        # Construct minimal polynomial of beta over GF(2)
        # Degree is 8 (since 17 is prime and 2^8 = 256, 2^d | 17-1=16 => d=4? No.
        # Order of 2 mod 17: 2^1=2, 2^2=4, 2^3=8, 2^4=16=-1, 2^8=1. So degree is 8.
        # Roots: beta, beta^2, beta^4, ..., beta^(2^7)
        
        # P(x) = (x - beta) (x - beta^2) ... (x - beta^(2^7))
        # Coefficients are in GF(2)
        
        poly_coeffs = [1] # Start with 1
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            # Multiply by (x + current_beta)
            # new_coeffs[i] = old_coeffs[i-1] + old_coeffs[i]*current_beta
            # Note: in GF(2), subtraction is addition.
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] ^= poly_coeffs[i]
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)
            
        # Check if coefficients are 0 or 1
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
            # Generator polynomial g(x) = sum g_i x^i
            # Codeword c(x) = u(x) g(x) mod (x^17 - 1) ? No, systematic or not?
            # Standard cyclic code: c(x) = u(x) g(x) where deg u < n-k = 17-9=8?
            # Wait, n=17, k=9. deg g = n-k = 8.
            # So u(x) has degree < 9.
            # c(x) = u(x) g(x). Since deg u < 9 and deg g = 8, deg c < 17.
            # So no wrap-around needed if we just multiply.
            
            codewords_17 = []
            for u in range(256): # 2^9
                c = 0
                # c(x) = sum_{j=0}^{8} u_j x^j * sum_{k=0}^{8} g_k x^k
                # c_m = sum_{j+k=m} u_j g_k
                # We can just do polynomial multiplication
                for j in range(9):
                    if (u >> j) & 1:
                        # Shift g_bits by j and XOR
                        for k in range(9):
                            if g_bits[k]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            # Extend to 18 bits
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            # Shorten (fix bit 17 to 0)
            # We keep codewords where the 18th bit is 0.
            # Since it's a linear code, exactly half will have 0.
            # This results in a [17, 8, 6] code?
            # Wait. Shortening a [18, 9, 6] code by fixing one coordinate to 0 gives a [17, 8, 6] code with 2^(9-1) = 128 words?
            # The task says "256-word linear reference is the [18,9,6] ... shortened by one coordinate".
            # If I fix one coordinate to 0, the dimension drops by 1. So 2^8 = 256? No, 2^(9-1)=128.
            # Let's re-read carefully: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate".
            # If the original is [18,9], size is 512. Shortening by 1 coordinate (fixing it) gives size 256.
            # Yes, because for every codeword with bit=1, there is a unique codeword with bit=0 (by linearity, add the codeword with only that bit set? No, that's not necessarily in the code).
            # Actually, shortening a linear code C of length n, dim k by fixing coordinate i to 0:
            # Let C' = { c in C | c_i = 0 }.
            # If the i-th coordinate of C is not identically 0, then dim(C') = k-1.
            # So size is 2^(k-1).
            # Here k=9, so size 256.
            # So we just filter the 512 extended codewords to keep those with bit 17 == 0.
            
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & MASK)
            
            # Remove duplicates (shouldn't be any, but safe)
            code_set = set(shortened_codewords)
            
            # Verify min distance
            # It should be at least 6.
            # Let's do a quick check on a small subset or trust the math.
            # Actually, let's just verify the whole set if it's fast enough.
            # 256 * 256 / 2 ~ 32k checks. Fast.
            min_d = 100
            lst = list(code_set)
            for i in range(len(lst)):
                for j in range(i+1, len(lst)):
                    d = (lst[i] ^ lst[j]).bit_count()
                    if d < min_d:
                        min_d = d
            if min_d >= 6:
                # We have a valid 256-word code.
                pass
            else:
                # Fallback if construction failed
                code_set = set()
    
    if not code_set:
        # Fallback: Greedy construction
        code_set.add(0)
        random.seed(42)
        candidates = list(range(1, 1 << 17))
        random.shuffle(candidates)
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            valid = True
            for w in code_set:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                code_set.add(cand)
        return list(code_set)

    # 2. Local Search to expand the code
    # Goal: Add words to reach 258 or more.
    # Strategy:
    # a. Try to add a random word. If it conflicts, try to "repair" by flipping bits of the conflicting existing words?
    #    No, flipping bits of existing words might break other distances.
    #    Better: Simulated Annealing on the set of codewords?
    #    Or: Greedy addition with random restarts of the "add" phase?
    #    
    #    Let's try a simple randomized greedy expansion first.
    #    1. Keep the 256 core words.
    #    2. Generate a pool of candidate words (e.g., all 2^17 is too big, but we can sample).
    #    3. For each candidate, check if it's valid. If yes, add it.
    #    4. If we get stuck, try to remove a word from the core and replace it with two new words? (Swap)
    #       This is complex.
    #    
    #    Let's try:
    #    1. Start with the 256 words.
    #    2. Create a list of all 2^17 words? No, 131k is manageable.
    #       Let's precompute the distance to the nearest codeword for all candidates?
    #       That's 131k * 256 operations ~ 33M ops. Might be too slow in Python.
    #       
    #    Alternative:
    #    1. Use a "conflict graph" approach.
    #    2. Try to add random words.
    #    3. If we can't add more, try to swap: remove one word, add two.
    #    
    #    Let's try a simpler approach:
    #    1. Iterate through all 2^17 words in random order.
    #    2. If a word is valid, add it.
    #    3. This is O(2^17 * |code|). 131k * 256 ~ 33M bit_count ops.
    #       bit_count is fast. 33M ops in Python might take ~10-30s.
    #       We have 80s. So we can do it.
    #       
    #    Let's optimize the validity check.
    #    Instead of checking against all words in the set, we can maintain a "blocked" set?
    #    No, because adding a word blocks its neighbors, but we don't want to remove words.
    #    
    #    Let's just do the random scan.
    
    current_code = set(code_set)
    
    # Precompute a list of all possible words? No, just iterate.
    # To make it deterministic and fast, we can use a fixed random seed.
    random.seed(12345)
    
    # We will try to add words.
    # We'll create a list of all candidates not in current_code.
    # Actually, iterating 0 to 2^17-1 is fine.
    
    # Optimization:
    # We can check candidates in batches.
    # But simple loop is okay.
    
    # Let's try to add as many as possible.
    # We'll do multiple passes if we have time.
    
    added_count = 0
    # We'll iterate through all 2^17 words.
    # To avoid bias, we shuffle the order.
    all_words = list(range(1 << 17))
    random.shuffle(all_words)
    
    # We only care about words not in current_code.
    # But checking `if cand in current_code` is O(1).
    
    for cand in all_words:
        if time.time() > deadline - 5.0:
            break
            
        if cand in current_code:
            continue
            
        # Check if cand is valid
        valid = True
        for w in current_code:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
                
        if valid:
            current_code.add(cand)
            added_count += 1
            # If we added a word, the set grew.
            # We don't need to restart the scan immediately, but we could.
            # Since we are scanning in random order, we'll just continue.
            
    # If we still have time, try a second pass?
    # The first pass should find all "easy" additions.
    # If we added some, maybe we can add more?
    # Unlikely, because the space is tight.
    
    # Let's try a "swap" strategy if we have time.
    # Remove one word, try to add two.
    # This is harder.
    
    # Let's just return what we have.
    # The goal is to beat 256.
    # If we find even 1 more, score is 257/32 = 8.03.
    # If we find 2, score is 258/32 = 8.06.
    
    return list(current_code)
