import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep margin

    # 1. GF(2^8) Arithmetic
    # Irreducible polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # Coeffs: 1 0 0 0 1 1 1 0 1
    # Binary: 100011101 = 285
    IRRED = 0b100011101
    
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

    # 2. Find a primitive 17th root of unity
    # x^17 - 1 = 0 => x^17 = 1
    # In GF(2^8), x^17 = 1 has 17 solutions.
    # We need one that is not 1.
    alpha = None
    for i in range(2, 256):
        if gf_pow(i, 17) == 1:
            if i != 1:
                alpha = i
                break
    
    if alpha is None:
        # Fallback to seed program if something goes wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Compute Minimal Polynomial g(x)
    # Roots are alpha^(2^j) for j=0..7
    # g(x) = prod_{j=0}^7 (x - alpha^(2^j))
    # In GF(2), x - a = x + a
    # We build the polynomial coefficients.
    # Start with 1
    poly = [1] # Coeffs of current product, index = power of x
    
    for j in range(8):
        root = gf_pow(alpha, 2**j)
        # Multiply poly by (x + root)
        # new_poly[k] = poly[k-1] + root * poly[k]
        new_poly = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            # Contribution to x^k from root * poly[k]
            new_poly[k] ^= gf_mul(root, poly[k])
            # Contribution to x^(k+1) from x * poly[k]
            new_poly[k+1] ^= poly[k]
        poly = new_poly
        
    # poly now has 9 coefficients: g_0 ... g_8
    # g(x) = sum_{j=0}^8 g_j x^j
    # Note: The constant term g_0 is the product of roots (with sign, which is + in GF2).
    # It should be 1.
    
    # 4. Generate Codewords
    # Basis vectors are x^i * g(x) for i=0..8
    # These are length 17.
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(9):
            vec[i+j] = poly[j]
        basis.append(vec)
        
    codewords = []
    # Iterate over all 512 combinations
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        
        # Extend with parity bit
        parity = sum(word) % 2
        # Append parity bit at position 17
        # The word is b_0 ... b_16, parity is b_17
        # Integer value: sum b_i 2^i
        val = 0
        for i in range(17):
            if word[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << 17)
            
        codewords.append(val)
        
    # Optional: Check time
    if time.time() > deadline - 5:
        # Return best effort, but this is deterministic so it should be fast
        pass
        
    return codewords
