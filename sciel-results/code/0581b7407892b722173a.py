import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # 1. GF(2^8) setup
    # Irreducible polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # Mask for reduction: 0x11D (binary 100011101)
    # Wait, x^8 = x^4 + x^3 + x^2 + 1.
    # So if bit 8 is set, we XOR with 0x11D (which is x^8 + x^4 + x^3 + x^2 + 1).
    # Actually, the mask for the lower 8 bits is 0xFF.
    # Reduction: if val >= 256, val ^= 0x11D << (val.bit_length() - 8) ?
    # Simpler: 
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
            b >>= 1
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # 2. Find alpha
    # beta = x (represented as 2)
    beta = 2
    alpha = gf_pow(beta, 15) # 255/17 = 15
    
    # 3. Compute generator polynomial g(x)
    # g(x) = prod_{i=0}^{7} (x + alpha^(2^i))
    # Represent polynomial as list of coefficients or int bitmask.
    # Let's use int bitmask where bit i is coeff of x^i.
    g = 1 # 1
    
    # We need to multiply g(x) by (x + alpha_i).
    # alpha_i is an element of GF(2^8).
    # But the result must have coefficients in GF(2).
    # This implies that the "alpha_i" part will cancel out or resolve to 0/1.
    # Actually, we can just do the multiplication in GF(2)[x] but treating the roots as symbols?
    # No, we can just evaluate the product directly.
    # Let's use the property that g(x) is the minimal polynomial.
    # We can find it by computing the sequence 1, alpha, alpha^2, ... and finding the linear dependence over GF(2).
    # But multiplication of factors is easier if we handle the coefficients correctly.
    # The coefficients of g(x) are in GF(2).
    # Let's just multiply the polynomials (x + r_1) ... (x + r_8).
    # The coefficients will be symmetric polynomials in the roots.
    # Since the roots are conjugates, the coefficients are in GF(2).
    
    # Let's try to compute g(x) by multiplying (x + alpha^(2^i))
    # We represent the polynomial as a list of GF(2^8) elements?
    # No, the final result is in GF(2)[x].
    # We can compute the coefficients one by one.
    # c_8 = 1
    # c_7 = sum(r_i)
    # c_6 = sum(r_i r_j)
    # ...
    # This is tedious.
    
    # Alternative: Just use the fact that g(x) is the minimal polynomial of alpha.
    # We can find it by trial:
    # Try polynomials of degree 8.
    # There are 2^8 = 256 polynomials of degree 8 with leading coeff 1.
    # We can iterate through all 256 candidates and check which one has alpha as a root.
    
    def gf_eval(poly, x):
        # poly is int bitmask
        # x is GF(2^8) element
        res = 0
        deg = poly.bit_length() - 1
        for i in range(deg + 1):
            if (poly >> i) & 1:
                res ^= gf_pow(x, i)
        return res

    # Find g(x)
    # g(x) must be monic of degree 8.
    # So g(x) = x^8 + c_7 x^7 + ... + c_0.
    # Bitmask: bit 8 is 1. Bits 0-7 are c_0...c_7.
    # We can iterate over all 256 possibilities for the lower 8 bits.
    
    g_poly = None
    for lower_bits in range(256):
        candidate = (1 << 8) | lower_bits
        if gf_eval(candidate, alpha) == 0:
            g_poly = candidate
            break
            
    if g_poly is None:
        # Fallback: return the seed code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 4. Generate codewords of the [17,9,5] code
    # Codewords are m(x) * g(x) mod (x^17 - 1).
    # m(x) is of degree < 9.
    # There are 512 such m(x).
    
    def poly_mul_mod(a, b, mod_poly):
        # a, b are int bitmasks
        # mod_poly is x^17 - 1, which is (1 << 17) ^ 1
        res = 0
        # Simple multiplication
        # Since degree is small, we can do it directly
        for i in range(18):
            if (a >> i) & 1:
                res ^= (b << i)
        # Reduce mod x^17 - 1
        # x^17 = 1
        # So terms with degree >= 17 wrap around.
        # res is at most degree 16+16=32? No, deg(a)<17, deg(b)<17.
        # Actually g(x) has degree 8. m(x) has degree < 9.
        # Product has degree < 17.
        # So no reduction needed if we just multiply!
        # Wait, m(x) g(x) has degree < 9+8=17.
        # So the product is just the standard polynomial product.
        return res

    words = []
    for m in range(512):
        c = poly_mul_mod(m, g_poly, 0) # No reduction needed
        # Check weight
        if c.bit_count() % 2 == 0:
            words.append(c)
            
    # The result should have 256 words.
    # Verify distance?
    # The task asks for a list.
    # We should ensure they are distinct.
    # Since it's a linear code, they are distinct.
    
    return words
