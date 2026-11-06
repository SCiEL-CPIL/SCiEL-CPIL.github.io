"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the [18, 9, 6] extended quadratic residue code.
    # This code has 512 codewords.
    
    # Field GF(2^8) setup
    # Irreducible polynomial: x^8 + x^4 + x^3 + x^2 + 1
    IRRED = 0b100011101 

    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:  # x^8
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

    # Find a primitive 17th root of unity alpha in GF(2^8)
    # The multiplicative group has order 255 = 15 * 17.
    # We need an element x such that x^17 = 1 and x != 1.
    alpha = None
    for x in range(2, 256):
        if gf_pow(x, 17) == 1:
            alpha = x
            break
            
    if alpha is None:
        # Fallback: This should not happen, but return a valid small code if it does
        # Using the Reed-Muller construction from the seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Cyclotomic coset C1 of 2 modulo 17
    # C1 = {1, 2, 4, 8, 9, 13, 15, 16}
    C1 = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Compute the generator polynomial g(x) = prod_{i in C1} (x + alpha^i)
    # In GF(2), x + alpha^i is represented by integer (2 | alpha^i)
    g = 1
    for i in C1:
        ai = gf_pow(alpha, i)
        g = gf_mul(g, 2 | ai)
        
    # g is a polynomial of degree 8 (fits in 9 bits)
    # Generate the 9 rows of the 9x17 generator matrix for the QR(17) code
    # Row k corresponds to x^k * g(x) mod (x^17 - 1)
    rows = []
    for k in range(9):
        row = 0
        for j in range(9):
            if (g >> j) & 1:
                pos = (j + k) % 17
                row |= (1 << pos)
        rows.append(row)
        
    # Generate all 512 codewords
    words = []
    for mask in range(512):
        w = 0
        for k in range(9):
            if (mask >> k) & 1:
                w ^= rows[k]
        # Extend with parity bit to get length 18
        p = w.bit_count() % 2
        full_w = w | (p << 17)
        words.append(full_w)
        
    return words
# EVOLVE-BLOCK-END
