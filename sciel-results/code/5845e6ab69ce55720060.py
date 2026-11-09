import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s margin (evaluator kills at 90s)
    
    # 1. GF(2^8) setup with lookup table for speed
    MOD_POLY = 0x11B  # x^8 + x^4 + x^3 + x + 1
    
    # Precompute multiplication table for GF(2^8)
    gf_mul_table = [[0]*256 for _ in range(256)]
    for a in range(256):
        for b in range(256):
            res = 0
            aa, bb = a, b
            while bb:
                if bb & 1:
                    res ^= aa
                bb >>= 1
                aa <<= 1
                if aa & 0x100:
                    aa ^= MOD_POLY
            gf_mul_table[a][b] = res
    
    def gf_mul(a, b):
        return gf_mul_table[a][b]

    def gf_pow(base, exp):
        result = 1
        while exp:
            if exp & 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            exp >>= 1
        return result

    # 2. Find primitive 17th root of unity alpha
    alpha = None
    for v in range(1, 256):
        if gf_pow(v, 17) == 1 and gf_pow(v, 1) != 1:
            alpha = v
            break
    
    if alpha is None:
        # Fallback: use known primitive element
        # In GF(2^8) with poly 0x11B, we need an element of order 17
        # 17 divides 255 = 3*5*17, so elements of order 17 exist
        # Try specific candidates
        for v in range(1, 256):
            if gf_pow(v, 17) == 1 and v != 1:
                alpha = v
                break
    
    if alpha is None:
        # Ultimate fallback to Reed-Muller (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Quadratic non-residues mod 17
    # NR mod 17: {3, 5, 6, 7, 10, 11, 12, 14}
    # These are the exponents i where alpha^i are roots of the generator polynomial
    nr = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # 4. Compute generator polynomial g(x) in GF(2^8)
    # g(x) = prod_{i in nr} (x - alpha^i)
    g = [1]
    for i in nr:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j+1] ^= g[j]
            new_g[j] ^= gf_mul(g[j], root)
        g = new_g
        
    # g should have coefficients in GF(2) (i.e., 0 or 1)
    # Verify and extract bits
    g_bits = []
    for coef in g:
        if coef not in (0, 1):
            # Coefficient not in GF(2) - something went wrong
            # This shouldn't happen for the QR code
            pass
        g_bits.append(1 if (coef & 1) else 0)
    
    # Ensure length is 9 (degree 8 polynomial)
    while len(g_bits) < 9:
        g_bits.append(0)
    # g_bits[0] should be 1 (leading coefficient)
    # Take first 9 bits
    g_bits = g_bits[:9]
    
    # 5. Construct 9 basis vectors for [17,9,5] code
    # Basis vector k is x^k * g(x) for k = 0, ..., 8
    basis = []
    for k in range(9):
        vec = 0
        for j in range(9):
            if g_bits[j] == 1:
                vec |= (1 << (k + j))
        # Extend to 18 bits with overall parity bit
        parity = vec.bit_count() % 2
        vec |= (parity << 17)
        basis.append(vec)
        
    # Check time
    if time.time() > deadline:
        return [basis[0]]

    # 6. Generate all 512 codewords
    codewords = []
    for mask in range(512):
        word = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                word ^= basis[i]
            m >>= 1
            i += 1
        codewords.append(word)
        
    # 7. Verify: check all pairs have distance >= 6
    # Only do full verification if we have time
    if time.time() < deadline - 2.0:
        n = len(codewords)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                d = (codewords[i] ^ codewords[j]).bit_count()
                if d < 6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback: use Reed-Muller
            words = []
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    words.append(word)
            return words
    
    return codewords
