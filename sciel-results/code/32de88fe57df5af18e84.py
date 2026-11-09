import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin

    # 1. Construct the [17,8,6] linear code
    # Generator polynomial for [18,9,6] QR code: g(x) = x^9 + x^7 + x^6 + x^5 + x^2 + x + 1
    # Coefficients: [1, 1, 0, 1, 1, 1, 0, 1, 0, 1] (x^9 down to x^0)
    g_poly = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
    
    def gf2_poly_mult(p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            if pi:
                for j, qj in enumerate(q):
                    r[i+j] ^= qj
        return r

    codewords_18 = []
    for msg in range(512):
        m_poly = [(msg >> i) & 1 for i in range(9)]
        c_poly = gf2_poly_mult(m_poly, g_poly)
        while len(c_poly) < 18:
            c_poly.append(0)
        codewords_18.append(c_poly)
        
    # Shorten by keeping codewords with last bit (index 17) = 0
    codewords_17 = []
    for c in codewords_18:
        if c[17] == 0:
            codewords_17.append(c[:17])
            
    # Convert to integers
    words = []
    for c in codewords_17:
        w = 0
        for i, bit in enumerate(c):
            if bit:
                w |= (1 << i)
        words.append(w)
        
    # Ensure distinct and sort for determinism
    words = sorted(list(set(words)))
    
    # 2. Greedy extension
    current_words = words[:]
    existing = set(current_words)
    
    # We iterate through all possible 17-bit integers
    # Optimization: If the current set is large, checking all candidates is expensive.
    # But 256 is small.
    
    for candidate in range(1 << 17):
        if time.time() > deadline:
            break
            
        if candidate in existing:
            continue
            
        # Check distance to all current words
        valid = True
        for w in current_words:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
                
        if valid:
            current_words.append(candidate)
            existing.add(candidate)
            
    return current_words
