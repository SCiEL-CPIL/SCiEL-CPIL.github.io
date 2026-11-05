import numpy as np

def construct():
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    poly = 0x11D
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return p

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            n >>= 1
        return result

    # Find a primitive element
    # We need an element alpha such that its order is 255.
    # Then beta = alpha^15 will have order 17.
    # Let's test alpha = 3
    alpha = 3
    # Check if alpha is primitive
    # alpha^255 == 1 and alpha^(255/p) != 1 for p in {3, 5, 17}
    is_primitive = True
    if gf_pow(alpha, 255) != 1:
        is_primitive = False
    else:
        for p in [3, 5, 17]:
            if gf_pow(alpha, 255 // p) == 1:
                is_primitive = False
                break
    
    if not is_primitive:
        # Find a primitive element
        for a in range(2, 256):
            if gf_pow(a, 255) != 1: continue
            ok = True
            for p in [3, 5, 17]:
                if gf_pow(a, 255 // p) == 1:
                    ok = False
                    break
            if ok:
                alpha = a
                break
    
    beta = gf_pow(alpha, 15)
    
    # Verify beta^17 == 1
    # assert gf_pow(beta, 17) == 1
    # assert beta != 1
    
    # Compute minimal polynomial of beta
    # Roots are beta^(2^i) for i=0..7
    p = [1] # p(x) = 1
    
    for i in range(8):
        r = gf_pow(beta, (1 << i))
        # Multiply p(x) by (x + r)
        new_p = [0] * (len(p) + 1)
        for j in range(len(p)):
            new_p[j] ^= gf_mul(p[j], r)
            new_p[j+1] ^= p[j]
        p = new_p
        
    # p is [c0, c1, ..., c8]
    # g(x) = p[0] + p[1]x + ... + p[8]x^8
    # Note: p[8] should be 1
    
    codewords = []
    seen = set()
    
    for msg in range(256):
        m = [0] * 9
        for i in range(9):
            if (msg >> i) & 1:
                m[i] = 1
        
        h = [0] * 17
        for i in range(9):
            if m[i] == 1:
                for j in range(9):
                    if p[j] == 1:
                        h[i+j] ^= 1
        
        word = 0
        for i in range(17):
            if h[i]:
                word |= (1 << i)
        
        # The [18,9,6] code is the extension.
        # We shorten by fixing the parity bit to 0.
        # This corresponds to keeping words with even weight in the [17,9,5] code.
        if word.bit_count() % 2 == 0:
            if word not in seen:
                seen.add(word)
                codewords.append(word)
                
    # Now try to add more words greedily
    # To speed up, we can check candidates in a specific order
    # or just iterate all
    
    # Optimization: Pre-compute distances? No, too much memory.
    # Just iterate.
    
    # We can use a list of current words for faster access
    current_words = codewords[:]
    
    # Iterate through all possible 17-bit words
    # To find nonlinear improvements, we should check all
    for candidate in range(2**17):
        if candidate in seen:
            continue
            
        valid = True
        for w in current_words:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        
        if valid:
            current_words.append(candidate)
            seen.add(candidate)
            
    return current_words
