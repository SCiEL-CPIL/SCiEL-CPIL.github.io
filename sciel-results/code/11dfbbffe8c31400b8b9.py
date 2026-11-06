import numpy as np

def construct():
    # GF(2^8) arithmetic
    # Irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
    def gf_mult(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11B
        return p

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mult(result, a)
            a = gf_mult(a, a)
            n >>= 1
        return result

    # Primitive element alpha = 2 (0x02)
    alpha = 0x02
    
    # Beta is a primitive 17th root of unity
    # alpha has order 255. 255 / 17 = 15.
    beta = gf_pow(alpha, 15)
    
    # Quadratic residues modulo 17
    R = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Compute generator polynomial g(x) = product (x + beta^r)
    # g(x) is in GF(2)[x]
    # Start with 1
    # We represent polynomials as lists of coefficients [c0, c1, ..., ck]
    # where c_i is in GF(2) (0 or 1)
    
    g_coeffs = [1] # 1
    
    for r in R:
        root = gf_pow(beta, r)
        # Multiply current g(x) by (x + root)
        # (x + root) has coeffs [root, 1] in GF(2^8)
        # But we are multiplying in GF(2^8)[x]
        # The result will have coefficients in GF(2^8)
        # However, since the set of roots is a cyclotomic coset, the final coefficients will be in GF(2)
        
        new_coeffs = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            # g_i * root + g_{i+1} * 1
            # Note: (x + root) * sum g_i x^i = sum g_i x^{i+1} + root * sum g_i x^i
            # coeff of x^k in product is g_{k-1} + root * g_k
            # Let's do it carefully.
            pass
            
        # Easier:
        # Product P = G * (x + root)
        # P_k = G_{k-1} + root * G_k
        # G_{-1} = 0, G_{len} = 0
        
        new_coeffs = [0] * (len(g_coeffs) + 1)
        for k in range(len(new_coeffs)):
            val = 0
            if k > 0:
                val ^= g_coeffs[k-1] # This is wrong, g_coeffs are in GF(2) but we are multiplying by root in GF(2^8)
                # Wait, g_coeffs are currently in GF(2^8) during intermediate steps?
                # No, we should keep them in GF(2^8) until the end.
                pass
        
        # Let's restart the polynomial multiplication logic properly.
        # g_coeffs are in GF(2^8).
        # Initially g = 1, so g_coeffs = [1] (1 in GF(2^8))
        
        # Actually, let's just use the property that g(x) is the minimal polynomial of beta.
        # We can find the minimal polynomial by finding the smallest m such that beta^(2^m) = beta.
        # But we know the degree is 8.
        # g(x) = (x + beta)(x + beta^2)...(x + beta^16)
        
        # Let's perform the multiplication in GF(2^8)[x]
        # Represent polynomial as list of ints (GF(2^8) elements)
        
        g = [1]
        for r in R:
            root = gf_pow(beta, r)
            # Multiply by (x + root)
            # new_g[k] = g[k-1] + root * g[k]
            new_g = [0] * (len(g) + 1)
            for k in range(len(new_g)):
                val = 0
                if k > 0:
                    val ^= g[k-1]
                if k < len(g):
                    val ^= gf_mult(root, g[k])
                new_g[k] = val
            g = new_g
        
        # After multiplying all factors, g should have coefficients in GF(2)
        # i.e., each g[i] should be 0 or 1.
        
    # g is a list of 9 coefficients [g0, ..., g8] in GF(2)
    # g0 is constant term.
    
    # Construct Generator Matrix G (9x17)
    # Row i is x^i * g(x) mod (x^17 - 1)
    # Since deg(g) = 8, x^i g(x) has degree i+8 <= 16 for i<=8.
    # So no wrap-around needed for i=0..8.
    
    G = np.zeros((9, 17), dtype=int)
    for i in range(9):
        # Shift g by i positions
        # g is [g0, g1, ..., g8]
        # x^i g(x) = g0 x^i + g1 x^{i+1} + ... + g8 x^{i+8}
        for j in range(9):
            G[i, i+j] = g[j]
            
    # Generate all 512 codewords
    codewords = []
    for mask in range(512):
        # Linear combination of rows of G
        word = np.zeros(17, dtype=int)
        for i in range(9):
            if (mask >> i) & 1:
                word ^= G[i]
        
        # Calculate parity bit
        parity = np.sum(word) % 2
        # Append parity bit
        word_18 = np.append(word, parity)
        
        # Convert to integer
        val = 0
        for bit in word_18:
            val = (val << 1) | bit
            
        codewords.append(val)
        
    return codewords
