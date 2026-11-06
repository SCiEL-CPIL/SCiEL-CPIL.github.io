import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    
    # Step 1: Construct the [17, 8, 6] linear code (256 words)
    # Using the GF(2^8) method from the seed program, which is verified to work.
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11d
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    beta = 3
    alpha = gf_pow(beta, 15)
    a3 = gf_pow(alpha, 3)
    
    v = [1] * 9
    for i in range(1, 9):
        v[i] = gf_mul(v[i-1], a3)
        
    M = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            M[j][i] = (v[i] >> j) & 1
            
    b = [(v[8] >> j) & 1 for j in range(8)]
    
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    
    for col in range(8):
        pivot = -1
        for row in range(col, 8):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(8):
            if row != col and A[row][col] == 1:
                for k in range(col, 9):
                    A[row][k] ^= A[col][k]
    
    c = [A[i][8] for i in range(8)]
    
    m3_poly = 0
    for i in range(8):
        if c[i]:
            m3_poly |= (1 << i)
    m3_poly |= (1 << 8)
    
    def poly_mul(x, y):
        res = 0
        while y:
            if y & 1:
                res ^= x
            y >>= 1
            x <<= 1
        return res
        
    x_plus_1 = 0b11
    h_poly = poly_mul(x_plus_1, m3_poly)
    
    def poly_mod(p):
        max_bit = p.bit_length()
        for i in range(max_bit - 1, 16, -1):
            if (p >> i) & 1:
                p ^= (1 << (i - 17))
        return p

    code_list = []
    code_set = set()
    for a in range(256):
        c_word = poly_mod(poly_mul(a, h_poly))
        c_word &= (1 << 17) - 1
        if c_word not in code_set:
            code_set.add(c_word)
            code_list.append(c_word)
            
    # Step 2: Try to add more words
    # We iterate through all possible 17-bit words.
    # To speed up, we can check candidates in a specific order or just all of them.
    # Given 2^17 is small, we check all.
    
    # Optimization: 
    # Instead of checking all 131k words against 256 codewords (which is ~33M ops),
    # we can be smarter. But 33M simple ops in Python might take ~10-20 seconds.
    # Let's try to optimize the inner loop.
    
    # We can precompute the bit-reversed or just use the list.
    # Let's add a time check in the loop.
    
    for w in range(1 << N):
        if time.time() > deadline:
            break
        if w in code_set:
            continue
            
        # Check distance to all existing codewords
        # Early exit if any distance < 6
        valid = True
        for c in code_list:
            if (w ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            code_set.add(w)
            code_list.append(w)
            
    return code_list
