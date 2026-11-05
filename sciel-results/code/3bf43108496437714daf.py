import numpy as np

def construct():
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    # This is the AES polynomial.
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

    # Find primitive element
    # Try 3
    alpha = 3
    # Check if alpha is primitive
    # Order should be 255
    # We need an element of order 17.
    # beta = alpha^15
    
    beta = gf_pow(alpha, 15)
    
    # Verify beta^17 == 1
    # beta17 = gf_pow(beta, 17)
    # assert beta17 == 1
    
    # Compute minimal polynomial of beta
    # Roots are beta^(2^i) for i=0..7
    # m(x) = prod (x - beta^(2^i))
    # In GF(2), subtraction is addition.
    # m(x) = x^8 + c7 x^7 + ... + c0
    
    # We can compute the coefficients by building the polynomial.
    # Start with p(x) = x
    # For each root r, p(x) = p(x) * (x + r)
    
    # p is a list of coefficients [c0, c1, ..., cn]
    p = [1] # p(x) = 1
    
    for i in range(8):
        r = gf_pow(beta, (1 << i))
        # Multiply p(x) by (x + r)
        # new_p[j] = p[j-1] + r*p[j]
        new_p = [0] * (len(p) + 1)
        for j in range(len(p)):
            new_p[j] ^= gf_mul(p[j], r)
            new_p[j+1] ^= p[j]
        p = new_p
        
    # p is [c0, c1, ..., c8] where c8=1
    # The generator polynomial g(x) = x^8 + c7 x^7 + ... + c0
    # In standard form for cyclic codes, we often use g(x) = 1 + c1 x + ... + c8 x^8
    # Let's stick to the list p where p[0] is constant term.
    
    # Generate codewords
    # A codeword is a multiple of g(x) modulo x^17 + 1.
    # c(x) = m(x) * g(x) mod (x^17 + 1)
    # m(x) has degree < 9.
    
    codewords = []
    
    for msg in range(256):
        # m(x) coefficients
        m = [0] * 9
        for i in range(9):
            if (msg >> i) & 1:
                m[i] = 1
        
        # Multiply m(x) by g(x)
        # g(x) coeffs are p[8], p[7], ..., p[0] if we align powers?
        # p is [c0, c1, ..., c8] for x^0 ... x^8.
        # g(x) = c0 + c1 x + ... + c8 x^8.
        
        # Product h(x) = m(x) g(x)
        # Degree of h is at most 8 + 8 = 16.
        # So we don't need to reduce modulo x^17 + 1 for the multiplication itself,
        # but we do if the degree goes over 16.
        # Max degree 16, so it fits in 17 bits.
        
        h = [0] * 17
        for i in range(9):
            if m[i] == 1:
                for j in range(9):
                    if p[j] == 1:
                        h[i+j] ^= 1
        
        # h is the codeword of length 17
        # Convert to integer
        word = 0
        for i in range(17):
            if h[i]:
                word |= (1 << i)
        
        # Extend with parity bit
        parity = word.bit_count() % 2
        word_ext = (word << 1) | parity
        
        # Shorten by fixing the last bit (parity bit) to 0
        # So we only keep words where parity == 0
        if parity == 0:
            codewords.append(word)
            
    return codewords
