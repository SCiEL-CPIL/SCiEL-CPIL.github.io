import time

def construct():
    """
    Constructs the extended binary quadratic residue code of length 18.
    This is a [18, 9, 6] code with 512 codewords.
    """
    # GF(2^8) setup using primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    poly = 0x11D
    
    # Precompute multiplication table for speed
    # gf_table[a][b] = a * b in GF(2^8)
    # This takes 256*256 operations, which is fast and one-time.
    gf_table = [[0]*256 for _ in range(256)]
    for a in range(256):
        for b in range(256):
            res = 0
            tmp_a = a
            tmp_b = b
            while tmp_b:
                if tmp_b & 1:
                    res ^= tmp_a
                tmp_b >>= 1
                tmp_a <<= 1
                if tmp_a & 0x100:
                    tmp_a ^= poly
            gf_table[a][b] = res

    def gf_mult(a, b):
        return gf_table[a][b]

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mult(res, a)
            a = gf_mult(a, a)
            n >>= 1
        return res

    # Find beta of order 17
    # 2^8 = 256 = 15*17 + 1
    # We need an element in GF(2^8) with order 17.
    # Since 17 is prime, any element != 1 with order dividing 17 must have order 17.
    # We check candidates.
    beta = None
    for candidate in range(2, 256):
        if gf_pow(candidate, 17) == 1:
            beta = candidate
            break
            
    if beta is None:
        # Fallback: Return a small valid code (Reed-Muller RM(1,4) extended? No, just return 0)
        # But this should not happen.
        return [0]

    # Roots: beta^1, beta^2, ..., beta^8 (conjugates)
    # Actually, the generator polynomial is the minimal polynomial of beta.
    # g(x) = prod_{i=0}^{7} (x - beta^{2^i})
    # We compute g(x) coefficients.
    # g(x) = x^8 + g7 x^7 + ... + g0
    # We represent g as a list of coefficients [g0, g1, ..., g8] where g8=1.
    g = [0] * 9
    g[8] = 1
    
    # Multiply by (x - root) iteratively
    # Start with poly = 1
    # poly = 1
    # for r in roots: poly = poly * (x + r)  (since char 2, - = +)
    
    current_poly = [1]
    for i in range(8):
        r = gf_pow(beta, 2**i)
        # Multiply current_poly by (x + r)
        new_poly = [0] * (len(current_poly) + 1)
        for j in range(len(current_poly)):
            # x * current_poly[j] -> degree j+1
            new_poly[j+1] ^= current_poly[j]
            # r * current_poly[j] -> degree j
            new_poly[j] ^= gf_mult(r, current_poly[j])
        current_poly = new_poly
        
    # current_poly should be length 9
    # g[0]..g[8]
    g = current_poly

    # Generate all 512 codewords
    # The code is the set of polynomials c(x) = m(x) * g(x) mod (x^17 - 1)
    # where m(x) is of degree < 9.
    # However, since g(x) is the generator of the cyclic code, the codewords are simply
    # the linear combinations of the basis vectors: g(x), x*g(x), ..., x^8*g(x) mod (x^17-1).
    
    # Let's generate the basis vectors as 17-bit masks.
    basis = []
    
    # g(x) as mask
    g_mask = 0
    for i in range(9):
        if g[i]:
            g_mask |= (1 << i)
            
    current = g_mask
    for i in range(9):
        basis.append(current)
        # Multiply by x mod (x^17 - 1)
        # Shift left by 1.
        # If bit 16 was set, it wraps to bit 0.
        # (current << 1) & 0x1FFFF
        # If (current & (1<<16)) != 0, then we need to XOR 1 (which is x^17 = 1)
        # Wait, x^17 = 1. So x * x^16 = x^17 = 1.
        # So if bit 16 is set, shift brings it out, we XOR 1 into bit 0.
        
        next_mask = (current << 1) & 0x1FFFF
        if current & (1 << 16):
            next_mask ^= 1
        current = next_mask

    words = []
    for m in range(512):
        w = 0
        for i in range(9):
            if (m >> i) & 1:
                w ^= basis[i]
        
        # Extend to 18 bits with parity
        parity = w.bit_count() % 2
        w_ext = w | (parity << 17)
        words.append(w_ext)
        
    return words
