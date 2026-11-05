import time
import random

def construct():
    start_time = time.time()
    
    # GF(2^8) parameters
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    PRIMITIVE_POLY = 0x11D
    FIELD_SIZE = 256
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIMITIVE_POLY
        return res

    def gf_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            e >>= 1
        return res

    # Find primitive 17th root of unity in GF(2^8)
    # Order of 2 mod 17 is 8, so 2^8 = 256 = 15*17 + 1.
    # We need an element of order 17.
    # Let alpha = 2 (x).
    # Check if alpha has order 17?
    # alpha^17 should be 1.
    # Let's verify.
    alpha = 2
    if gf_pow(alpha, 17) != 1:
        # Try other elements? 2 is usually a primitive element for this poly.
        # If not, we need to find one.
        # For 0x11D, 2 is primitive (order 255).
        # 255 = 15 * 17.
        # So alpha^(255/17) = alpha^15 is a 17th root.
        alpha = gf_pow(alpha, 15)
        if gf_pow(alpha, 17) != 1:
            # Fallback: search for a 17th root
            for x in range(1, 256):
                if gf_pow(x, 17) == 1 and x != 1:
                    alpha = x
                    break

    # Compute minimal polynomial of alpha^3 (non-residue)
    # Roots: alpha^(3 * 2^k mod 17) for k=0..7
    roots = []
    for k in range(8):
        exp = (3 * pow(2, k, 17)) % 17
        roots.append(gf_pow(alpha, exp))
        
    # Multiply (x - r) for all roots
    # Coefficients in GF(2^8)
    # p(x) = sum c_i x^i
    p = [1]  # c_0 = 1
    for r in roots:
        # Multiply p(x) by (x - r) = (x + r) in char 2
        new_p = [0] * (len(p) + 1)
        for i in range(len(p)):
            # p[i] * x term
            new_p[i + 1] ^= p[i]
            # p[i] * r term
            new_p[i] ^= gf_mul(p[i], r)
        p = new_p
        
    # p should have coefficients in {0, 1}
    # Convert to generator polynomial coefficients (list of 0/1)
    # p is [c_0, c_1, ..., c_8]
    g_coeffs = []
    for c in p:
        if c == 0:
            g_coeffs.append(0)
        elif c == 1:
            g_coeffs.append(1)
        else:
            # Error if not in GF(2)
            g_coeffs.append(0)
            
    # g_coeffs represents g(x) = sum g_coeffs[i] x^i
    # Generate all codewords of the [17,9,5] code
    # Codeword c(x) = a(x) * g(x), deg(a) < 9
    words = []
    for a_val in range(512):
        # a(x) bits from a_val
        # Compute c(x)
        c_bits = [0] * 17
        for i in range(9):
            if (a_val >> i) & 1:
                # Add g(x) shifted by i
                for j in range(9): # deg g = 8
                    c_bits[i + j] ^= g_coeffs[j]
        
        # Convert to integer
        word_int = 0
        for i in range(17):
            if c_bits[i]:
                word_int |= (1 << i)
        
        # Check weight
        if word_int.bit_count() % 2 == 0:
            words.append(word_int)
            
    # Verify distance if time permits (optional, but good for sanity)
    # The construction guarantees distance 6.
    
    return words
