import random
import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 45.0

    # GF(256) implementation
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # a(x) = x^8 + x^4 + x^3 + x^2 + 1
    # We represent polynomials as integers (bitmasks).
    
    def gf256_mul(a, b):
        res = 0
        for i in range(8):
            if b & (1 << i):
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11D # x^8 + x^4 + x^3 + x^2 + 1
        return res & 0xFF

    # Find primitive element alpha
    # alpha is a root of the primitive polynomial? No, alpha is the element x in GF(256).
    # Let alpha = 2 (x).
    # We need the minimal polynomial of alpha.
    # We can compute it by finding the smallest degree polynomial m(x) such that m(alpha) = 0.
    
    # Alternatively, we know the roots are alpha^1, alpha^2, ...
    # We can construct g(x) directly.
    # g(x) = prod_{i in {0, 1, 2, 4, 8, 16, 15, 13, 9}} (x + alpha^i)
    # Note: x+alpha^i is the same as x-alpha^i in GF(2).
    
    # Let's compute g(x) coefficients.
    # g(x) = x^9 + ... + g0
    
    # Roots:
    # 0 (from x+1)
    # 1, 2, 4, 8, 16, 15, 13, 9 (from m1)
    
    # We need alpha values.
    # alpha = 2
    # alpha^1 = 2
    # alpha^2 = 4
    # alpha^3 = 8
    # alpha^4 = 16
    # alpha^5 = 32
    # alpha^6 = 64
    # alpha^7 = 128
    # alpha^8 = 128*2 = 256 = 256 - 255 + 29 = 29? 
    # 256 mod (x^8+x^4+x^3+x^2+1)
    # x^8 = x^4+x^3+x^2+1
    # So alpha^8 = 16 + 8 + 4 + 1 = 29.
    
    # Let's compute powers of alpha properly.
    alpha = 2
    powers = [1]
    for i in range(1, 17):
        powers.append(gf256_mul(powers[-1], 2))
        
    # Roots of g(x) are alpha^0, alpha^1, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9
    # Wait, alpha^0 = 1.
    # The roots are 1 (from x+1? No, x+1 has root 1).
    # So root 1 is included.
    # The other roots are alpha^1, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9.
    
    root_indices = [0, 1, 2, 4, 8, 16, 15, 13, 9]
    
    # Compute g(x) = prod (x + alpha^i)
    # Start with g(x) = 1
    g_coeffs = [1] # constant term
    
    for idx in root_indices:
        root = powers[idx]
        # Multiply g(x) by (x + root)
        # new_g = g * x + g * root
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= g_coeffs[i] * root # g * root
            new_g[i+1] ^= g_coeffs[i] # g * x
        g_coeffs = new_g
        
    # g_coeffs[0] is constant, g_coeffs[9] is x^9 coeff
    # g(x) = g_coeffs[9] x^9 + ... + g_coeffs[0]
    
    # Construct generator matrix G (8 rows, 17 cols)
    # Row i is coefficients of x^i g(x) mod (x^17 - 1)
    G = []
    for i in range(8):
        row = [0] * 17
        # Multiply g(x) by x^i
        # g(x) = sum_{k=0}^9 g_k x^k
        # x^i g(x) = sum_{k=0}^9 g_k x^{k+i}
        for k in range(10):
            coeff = g_coeffs[k]
            pos = k + i
            if pos < 17:
                row[pos] ^= coeff
            else:
                # Wrap around mod x^17 - 1
                # x^17 = 1
                row[pos - 17] ^= coeff
        G.append(row)
        
    # G is 8x17 matrix over GF(2)
    # Generate all 256 codewords
    base_words = []
    for mask in range(256):
        word = 0
        for i in range(8):
            if mask & (1 << i):
                # XOR row i into word
                for j in range(17):
                    if G[i][j]:
                        word ^= (1 << j)
        base_words.append(word)
        
    codewords = list(base_words)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    def hamming_dist(a, b):
        return bin(a ^ b).count('1')
        
    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
        
    rng = random.Random(42)
    
    # Phase 1: Greedy addition of random compatible words
    attempts = 0
    max_attempts_greedy = 500000
    while attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.4:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
        
    # Phase 2: Local search / Greedy clique on candidates
    # Generate candidates by flipping bits in existing codewords
    candidate_set = set()
    for c in codewords:
        # Flip 1-2 bits
        for mask in range(1, 1 << N):
            if bin(mask).count('1') > 2:
                continue
            w = c ^ mask
            if w not in candidate_set:
                candidate_set.add(w)
                
    # Filter valid candidates
    valid_candidates = []
    for w in candidate_set:
        if is_compatible_with_all(w, codewords):
            valid_candidates.append(w)
            
    # Greedy clique
    def greedy_clique(candidates, existing):
        if not candidates:
            return []
        clique = []
        remaining = list(candidates)
        while remaining:
            if time.time() - start_time > time_limit * 0.8:
                break
            # Pick first candidate (or one with high degree)
            # For speed, just pick first
            c = remaining.pop(0)
            clique.append(c)
            remaining = [other for other in remaining if hamming_dist(c, other) >= MIN_DIST]
        return clique
        
    if valid_candidates and time.time() - start_time < time_limit * 0.7:
        if len(valid_candidates) > 1000:
            valid_candidates = rng.sample(valid_candidates, 1000)
        clique = greedy_clique(valid_candidates, codewords)
        if clique:
            for w in clique:
                if is_compatible_with_all(w, codewords):
                    codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                
    # Phase 3: Final polish
    attempts = 0
    max_attempts_final = 500000
    while attempts < max_attempts_final:
        if time.time() - start_time > time_limit:
            break
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
        
    return best_codewords
