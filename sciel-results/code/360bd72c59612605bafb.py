import numpy as np

def construct():
        # 1. Find generator polynomial for [17,8,5] QR code
        # Primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
        # Represent elements of GF(2^8) as integers 0..255
        # alpha = 2 (since x is a root of p(x), x^8 = x^4+x^3+x^2+1)
        # Wait, if p(x) = x^8 + x^4 + x^3 + x^2 + 1, then alpha^8 = alpha^4 + alpha^3 + alpha^2 + 1.
        
        # We need to compute f_1(x) = prod (x - alpha^i) for i in {1,2,4,8,16,15,13,9}
        # This is equivalent to finding the minimal polynomial of alpha.
        # Since alpha is primitive, the minimal polynomial is degree 8.
        # We can compute it by finding a linear dependence among 1, alpha, ..., alpha^8.
        # Or just multiply (x - alpha^i) in GF(2)[x] using GF(2^8) arithmetic?
        # Easier: Use the fact that f_1(x) is a factor of x^17 - 1.
        # We can find it by GCD(x^17 - 1, x^(2^k) - x) ?
        
        # Actually, we can just hardcode the generator polynomial if we calculate it once.
        # But we should calculate it to be safe.
        
        # Let's compute the minimal polynomial of alpha (root of p(x)).
        # The minimal polynomial is the same as f_1(x) because the coset is the whole set of residues?
        # Yes, C1 is the set of residues.
        
        # Compute f_1(x)
        # f_1(x) = x^8 + a7 x^7 + ... + a0
        # We can find it by solving the system for the minimal polynomial.
        
        # Alternative:
        # The generator of the [17,8,5] code is g(x) = (x+1) f_1(x).
        # g(x) has degree 9.
        # We can find g(x) directly as the GCD of x^17-1 and ...?
        
        # Let's just compute f_1(x) using the product formula in GF(2)[x].
        # We need to do arithmetic in GF(2^8) to evaluate alpha^i?
        # No, we can compute the polynomial coefficients directly.
        
        # Helper to multiply polynomials over GF(2)
        def poly_mul(p1, p2):
            res = [0] * (len(p1) + len(p2) - 1)
            for i, c1 in enumerate(p1):
                if c1:
                    for j, c2 in enumerate(p2):
                        if c2:
                            res[i+j] ^= 1
            return res

        def poly_add(p1, p2):
            max_len = max(len(p1), len(p2))
            res = [0] * max_len
            for i in range(len(p1)):
                res[i] ^= p1[i]
            for i in range(len(p2)):
                res[i] ^= p2[i]
            return res

        # Compute f_1(x)
        # Roots: alpha^1, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9
        # We need alpha in GF(2^8).
        # Let's use a primitive polynomial.
        # p(x) = x^8 + x^4 + x^3 + x^2 + 1
        # This defines the field.
        
        # To compute the minimal polynomial, we can use the Berlekamp-Massey algorithm
        # on the sequence 1, alpha, alpha^2, ...
        # Or simply multiply (x - alpha^i) sequentially.
        # But we need to represent alpha^i.
        
        # Let's represent elements of GF(2^8) as integers.
        # Multiplication table or reduction.
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                # Reduce a modulo p(x) = x^8 + x^4 + x^3 + x^2 + 1
                # p(x) = 0x119? 10010011?
                # x^8 + x^4 + x^3 + x^2 + 1
                # Bits: 8, 4, 3, 2, 0.
                # Mask: 1<<8 | 1<<4 | 1<<3 | 1<<2 | 1<<0 = 256 + 16 + 8 + 4 + 1 = 285 = 0x11D
                if a & 0x100:
                    a ^= 0x11D
            return res

        # Compute alpha^i
        # alpha = 2 (x)
        # We need the minimal polynomial of alpha.
        # f_1(x) = prod (x - alpha^i) for i in C1.
        
        # We can compute f_1(x) by starting with [1, 0] (x) and multiplying by (x - alpha^i)?
        # No, (x - alpha) is not a polynomial over GF(2) with coefficients in GF(2).
        # The factors are over GF(2^8).
        # But the product is over GF(2).
        
        # Better approach:
        # f_1(x) is the minimal polynomial of alpha.
        # We can find it by finding the smallest k such that alpha^k is in the span of {1, ..., alpha^{k-1}}.
        # Since degree is 8, k=8.
        # alpha^8 = alpha^4 + alpha^3 + alpha^2 + 1 (from p(x)=0)
        # So x^8 + x^4 + x^3 + x^2 + 1 is the minimal polynomial?
        # Yes, if p(x) is irreducible.
        # So f_1(x) = x^8 + x^4 + x^3 + x^2 + 1.
        
        # Check if p(x) is irreducible.
        # It is a known primitive polynomial.
        
        # So f_1(x) = [1, 0, 0, 1, 1, 1, 0, 0, 1] (coeffs of x^0 to x^8)
        # Wait, x^8 + x^4 + x^3 + x^2 + 1
        # Coeffs: 1 (x^0), 0, 0, 1 (x^2), 1 (x^3), 1 (x^4), 0, 0, 1 (x^8)
        # List: [1, 0, 0, 1, 1, 1, 0, 0, 1]
        
        f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
        
        # g(x) = (x+1) f1(x)
        x_plus_1 = [1, 1]
        g = poly_mul(f1, x_plus_1)
        # g has degree 9.
        # g[0] ... g[9]
        
        # Generate basis for [17,8,5] code
        # Basis vectors are shifts of g(x)
        basis = []
        for i in range(8):
            vec = [0] * 17
            for j in range(len(g)):
                vec[i+j] = g[j]
            basis.append(vec)
            
        # Generate all 256 codewords
        words_17 = []
        for mask in range(256):
            word = [0] * 17
            for i in range(8):
                if mask & (1 << i):
                    for j in range(17):
                        word[j] ^= basis[i][j]
            words_17.append(word)
            
        # Extend to 18 bits
        words_18 = []
        for w in words_17:
            parity = sum(w) % 2
            words_18.append(w + [parity])
            
        # Shorten: keep words where bit 0 is 0
        # Note: "shortened by one coordinate" usually implies removing that coordinate.
        # We fix coordinate 0 to 0.
        # The resulting words are the remaining 17 bits.
        # Since we fixed bit 0 to 0, the remaining bits are w[1:18].
        # Wait, the original code C was length 17.
        # We extended it to 18.
        # We fix bit 0 (the first bit of the original 17) to 0.
        # So we take w[1:18] from the 18-bit word?
        # No, the 18-bit word is (c, parity).
        # c is 17 bits.
        # We fix c[0] = 0.
        # The codeword in the shortened code is c[1:17] (16 bits)?
        # No, shortening reduces length by 1.
        # Original length 18. Shortened length 17.
        # We remove coordinate 0.
        # The remaining coordinates are 1..17.
        # So the new word is (c[1], ..., c[16], parity).
        # This has length 17.
        
        final_words = []
        for w18 in words_18:
            if w18[0] == 0:
                # Remove index 0
                new_word = w18[1:]
                # Convert to integer
                val = 0
                for i, bit in enumerate(new_word):
                    if bit:
                        val |= (1 << i)
                final_words.append(val)
                
        return final_words
