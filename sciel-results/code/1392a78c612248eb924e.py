import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin

    # GF(2^8) parameters
    IRRED = 0b100011101 # x^8 + x^4 + x^3 + x^2 + 1
    
    # Precompute multiplication table for GF(2^8)
    # This is faster than bit manipulation in loops for many operations
    # But 256x256 table is 65536 entries. Generation might take time.
    # Let's stick to efficient bit manipulation or a smaller table if possible.
    # Actually, we only need to multiply by powers of alpha or specific roots.
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= IRRED
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find primitive 17th root of unity
    alpha = None
    for i in range(2, 256):
        if gf_pow(i, 17) == 1 and i != 1:
            # Check if it's primitive (order exactly 17)
            # 17 is prime, so any i != 1 with i^17=1 is primitive.
            alpha = i
            break

    if alpha is None:
        # Fallback: Reed-Muller [16, 5, 8] padded to 18
        # Size 32
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Pad with 2 zeros
                words.append(word)
        return words

    # Quadratic residues mod 17
    # 1, 2, 4, 8, 9, 13, 15, 16
    residues = [1, 2, 4, 8, 9, 13, 15, 16]
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]

    # We want the DUAL QR code, which has generator polynomial with roots = non-residues.
    # g_dual(x) = prod_{r in non_residues} (x - alpha^r)
    # In GF(2), x - alpha^r = x + alpha^r
    
    poly = [1] # 1
    
    for r in non_residues:
        root = gf_pow(alpha, r)
        # Multiply poly by (x + root)
        new_poly = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            new_poly[k] ^= gf_mul(root, poly[k])
            new_poly[k+1] ^= poly[k]
        poly = new_poly

    # poly has degree 8, so 9 coefficients.
    # This is the generator polynomial of the [17, 8, 6] code.
    
    # Generate codewords
    # Each codeword is c(x) = m(x) * g(x) mod (x^17 - 1)
    # Since we are in the cyclic code setting, we can just compute the product.
    # But wait, the code is cyclic.
    # We need to generate all 2^8 = 256 messages m(x) of degree < 8.
    # c(x) = m(x) g(x). Degree of c is at most 7+8=15.
    # So c(x) fits in 17 bits (indices 0..16).
    
    codewords = []
    for mask in range(256):
        # m(x) coefficients
        m_coeffs = [(mask >> i) & 1 for i in range(8)]
        
        # c(x) = sum m_i x^i * g(x)
        # We can precompute the basis vectors g(x) * x^i for i=0..7
        # But let's just do the convolution.
        
        c = [0] * 17
        for i in range(8):
            if m_coeffs[i]:
                # Add g(x) shifted by i
                for j in range(9):
                    c[i+j] ^= poly[j]
        
        # c is a 17-bit vector
        val = 0
        for i in range(17):
            if c[i]:
                val |= (1 << i)
        
        # Extend to 18 bits with parity
        parity = bin(val).count('1') % 2
        if parity:
            val |= (1 << 17)
            
        codewords.append(val)
        
    # Check time
    if time.time() > deadline:
         # If this took too long (unlikely for 256 words), return what we have
         pass
         
    return codewords
