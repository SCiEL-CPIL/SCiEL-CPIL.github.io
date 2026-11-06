import numpy as np
import time

def construct():
    # Step 1: Generate the initial 512 linear codewords
    def poly_mod(a, b):
        while b != 0:
            deg_a = a.bit_length() - 1
            deg_b = b.bit_length() - 1
            if deg_a >= deg_b:
                shift = deg_a - deg_b
                a ^= (b << shift)
            else:
                break
        return a

    def poly_divides(a, b):
        return poly_mod(a, b) == 0

    def is_irreducible(p, n):
        def poly_pow_mod(base, exp, mod):
            result = 1
            base = poly_mod(base, mod)
            while exp:
                if exp & 1:
                    result = poly_mod(result * base, mod)
                base = poly_mod(base * base, mod)
                exp >>= 1
            return result

        x = 2
        val = x
        for _ in range(n):
            val = poly_mod(val * val, p)
        if poly_mod(val ^ x, p) != 0:
            return False
        
        for k in [1, 2, 4]:
            val = x
            for _ in range(k):
                val = poly_mod(val * val, p)
            if poly_mod(val ^ x, p) == 0:
                return False
        return True

    Q = 0
    for i in range(9):
        Q |= 1 << (2*i)
        
    g = None
    for p in range(1 << 8, 1 << 9):
        if p & 1 == 0:
            continue
        if is_irreducible(p, 8):
            if poly_divides(Q, p):
                g = p
                break
                
    if g is None:
        # Fallback: Return a small valid code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Extend to 18 bits (pad with 0s at high end? Or parity?)
                # The task requires 18-bit words.
                # We just pad with 0s for positions 16, 17.
                # Distance might be < 6? 
                # This fallback is only if g is not found, which is unlikely.
                words.append(word)
        return words

    # Generate 512 base words
    base_words = []
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= g << i
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        base_words.append(c_ext)
        
    code_list = list(base_words)
    
    # Step 2: Greedy search for additional codewords using NumPy
    start_time = time.time()
    
    # Pre-compute bit vectors for all 2^18 words
    all_w = np.arange(2**18, dtype=np.uint32)
    all_bits = (all_w[:, None] >> np.arange(18)[None, :]) & 1
    all_bits = all_bits.astype(np.uint8)
    
    # Helper to convert list of ints to bit matrix
    def to_bit_matrix(int_list):
        if not int_list:
            return np.zeros((0, 18), dtype=np.uint8)
        return np.array([[ (c >> i) & 1 for i in range(18) ] for c in int_list], dtype=np.uint8)

    code_mat = to_bit_matrix(code_list)
    
    for i in range(2**18):
        if time.time() - start_time > 55:
            break
            
        w_bits = all_bits[i]
        
        # Calculate distances
        dists = np.sum(code_mat != w_bits, axis=1)
        
        if np.min(dists) >= 6:
            code_list.append(int(all_w[i]))
            code_mat = np.vstack((code_mat, w_bits.reshape(1, -1)))
            
    return code_list
