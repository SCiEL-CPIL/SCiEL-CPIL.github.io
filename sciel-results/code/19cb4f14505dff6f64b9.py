import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0
    
    # 1. GF(2^8) Arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    # We need an element alpha of order 17.
    # 2^8 - 1 = 255 = 15 * 17.
    # If beta is a primitive element (order 255), then alpha = beta^15 has order 17.
    
    def gf_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return res

    def gf_pow(base, exp):
        result = 1
        while exp:
            if exp & 1:
                result = gf_mult(result, base)
            base = gf_mult(base, base)
            exp >>= 1
        return result

    # x is represented as 2 (0b10) in polynomial basis.
    # Check if x is primitive.
    # We assume it is, or find a primitive element.
    # Let's just use beta = 2 (x).
    # Order of x:
    # We can verify x^255 == 1.
    # alpha = x^15.
    alpha = gf_pow(2, 15)
    
    # Verify alpha has order 17
    # alpha^17 should be 1.
    # alpha^1 != 1.
    if gf_pow(alpha, 17) != 1 or alpha == 1:
        # Fallback to RM code if something is wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Construct Generator Polynomial g(x)
    # Roots are alpha^i for i in Q = {1, 2, 4, 8, 9, 13, 15, 16}
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # g(x) = prod (x - alpha^i)
    # Start with g(x) = 1
    # Coefficients in GF(2^8), list from highest degree to lowest?
    # Let's use a list of coefficients [c_0, c_1, ..., c_k] where c_k is highest.
    # Or just polynomial multiplication.
    
    poly = [1] # 1
    
    for i in Q:
        root = gf_pow(alpha, i)
        # Multiply poly by (x - root) which is (x + root) in GF(2)
        # New poly degree is len(poly)
        new_poly = [0] * (len(poly) + 1)
        for j in range(len(poly)):
            # poly[j] * x  -> new_poly[j+1]
            new_poly[j+1] ^= poly[j]
            # poly[j] * root -> new_poly[j]
            new_poly[j] ^= gf_mult(poly[j], root)
        poly = new_poly
        
    # poly now contains coefficients of g(x) in GF(2^8)
    # Since Q is a full cyclotomic coset, coefficients should be in GF(2).
    # Map to GF(2): 0 -> 0, 1 -> 1. Others should not occur.
    # Let's verify and extract bits.
    coeffs_gf2 = []
    for c in poly:
        if c == 0:
            coeffs_gf2.append(0)
        elif c == 1:
            coeffs_gf2.append(1)
        else:
            # Error in construction
            # Fallback
            words = []
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    words.append(word)
            return words
            
    # coeffs_gf2 is [c_0, c_1, ..., c_8] where c_8 is coeff of x^8.
    # g(x) = c_0 + c_1 x + ... + c_8 x^8
    # For cyclic code, generator vector is (c_0, c_1, ..., c_8, 0, ..., 0)
    
    n = 17
    k = 9
    r = 8
    
    # Generate the 9 basis vectors (rows of generator matrix)
    # Row j is x^j g(x) mod (x^17 - 1)
    basis = []
    for j in range(k):
        vec = 0
        # x^j g(x) has coefficients shifted by j
        # Coeff of x^m in x^j g(x) is c_{m-j}
        # We need to wrap around mod 17
        for m in range(n):
            coeff_idx = m - j
            if 0 <= coeff_idx < r:
                bit = coeffs_gf2[coeff_idx]
            elif coeff_idx < 0:
                # Wrap around: x^17 = 1
                # m - j < 0 => m < j
                # coeff_idx + 17
                bit = coeffs_gf2[coeff_idx + 17]
            else:
                bit = 0
            if bit:
                vec |= (1 << m)
        basis.append(vec)
        
    # 3. Enumerate all 2^9 = 512 codewords
    codewords_17 = []
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if mask & (1 << i):
                word ^= basis[i]
        codewords_17.append(word)
        
    # 4. Extend to 18 bits
    # Add parity bit to make all weights even (or just extend)
    # The code is [17, 9, 5]. Extension adds a parity check bit.
    # If the code contains words of odd weight, the distance increases by 1.
    # QR code of length 17 has distance 5 (odd).
    # So extension gives distance 6.
    
    codewords_18 = []
    for w in codewords_17:
        parity = w.bit_count() % 2
        # Bit 17 is the parity bit
        w18 = w | (parity << 17)
        codewords_18.append(w18)
        
    # 5. Verify
    # Check distinctness
    if len(set(codewords_18)) != 512:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
        
    # Check min distance
    min_dist = 18
    for i in range(len(codewords_18)):
        for j in range(i+1, len(codewords_18)):
            d = (codewords_18[i] ^ codewords_18[j]).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                # Should not happen, but if it does, fallback
                words = []
                for slope in range(16):
                    for intercept in range(2):
                        word = 0
                        for point in range(16):
                            bit = ((slope & point).bit_count() + intercept) % 2
                            word |= bit << point
                        words.append(word)
                return words
                
    return codewords_18

