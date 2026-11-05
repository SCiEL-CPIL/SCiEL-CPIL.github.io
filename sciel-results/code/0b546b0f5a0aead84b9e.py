import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Define GF(2^8) arithmetic
    PRIMITIVE_POLY = 0x11D
    
    def gf_mult(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIMITIVE_POLY
            b >>= 1
        return p

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mult(result, a)
            a = gf_mult(a, a)
            n >>= 1
        return result

    # 2. Find a primitive 17th root of unity
    beta = 2
    zeta = gf_pow(beta, 15)
    
    # 3. Compute the generator polynomial g(x)
    roots_exponents = [1, 2, 4, 8, 16, 15, 13, 9]
    
    g_coeffs = [1]
    
    for exp in roots_exponents:
        if time.time() > deadline:
            break
        root = gf_pow(zeta, exp)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mult(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    g_poly = 0
    for i, c in enumerate(g_coeffs):
        if c == 1:
            g_poly |= (1 << i)

    # 4. Generate all 512 codewords
    codewords = []
    for msg in range(512):
        if time.time() > deadline:
            break
        c_word = 0
        temp_msg = msg
        msg_idx = 0
        while temp_msg:
            if temp_msg & 1:
                c_word ^= (g_poly << msg_idx)
            temp_msg >>= 1
            msg_idx += 1
            
        parity = c_word.bit_count() % 2
        full_word = c_word | (parity << 17)
        codewords.append(full_word)
        
    return codewords
