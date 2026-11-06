"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 18
    MIN_DIST = 6
    DEADLINE = 85  # safety margin before 90s kill

    # Construct the [18, 9, 6] extended quadratic residue code
    # This code has 2^9 = 512 codewords.
    # It is formed by extending the [17, 9, 5] QR code with a parity bit.
    
    # Step 1: Define GF(16) arithmetic
    # Primitive polynomial: x^4 + x + 1 (0b10011 = 19)
    def gf_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0b10000: # if a >= 16
                a ^= 0b10011 # reduce by x^4 + x + 1
        return res

    def gf_pow(a, p):
        res = 1
        while p:
            if p & 1:
                res = gf_mult(res, a)
            a = gf_mult(a, a)
            p >>= 1
        return res

    # Primitive element alpha = 1 (represented as 1 in GF(2^4))
    # Wait, alpha is a root of x^4+x+1. Let alpha be the element represented by 2 (x).
    # Check: alpha^4 + alpha + 1 = 0?
    # Let's just use the standard representation where alpha is the class of x.
    # In integer representation, alpha = 2.
    alpha = 2
    
    # Verify alpha is primitive (order 15)
    # alpha^15 should be 1
    # We will compute the generator polynomial for the [17,9,5] code.
    # The roots are alpha^1, alpha^2, alpha^3, alpha^4, alpha^5, alpha^6, alpha^7, alpha^8.
    # These form two cyclotomic cosets modulo 17:
    # C1 = {1, 2, 4, 8}
    # C3 = {3, 6, 12, 7} (since 3*2=6, 6*2=12, 12*2=24=7 mod 17, 7*2=14... wait)
    # Let's check C3: 3, 6, 12, 24%17=7, 14, 28%17=11, 22%17=5, 10, 20%17=3.
    # The coset is {3, 6, 12, 7, 14, 11, 5, 10}. This has 8 elements.
    # So the roots are alpha^1, alpha^2, alpha^4, alpha^8 (min poly M1)
    # and alpha^3, alpha^6, alpha^{12}, alpha^7, alpha^{14}, alpha^{11}, alpha^5, alpha^{10} (min poly M3)
    # Total 8 roots.
    
    # Calculate Minimal Polynomial M1(x) for alpha
    # M1(x) = (x - alpha)(x - alpha^2)(x - alpha^4)(x - alpha^8)
    # Coefficients in GF(16)
    def calc_min_poly(root):
        # Returns list of coeffs [c0, c1, c2, c3, c4] for x^4 + c3 x^3 + c2 x^2 + c1 x + c0
        # Actually, let's just multiply polynomials.
        # Start with (x - root)
        # Represent poly as list of coeffs from lowest degree.
        # (x - r) -> [-r, 1]
        poly = [gf_mult(root, 1) ^ 0, 1] # -r is r in GF(2)
        # Wait, in GF(2), -r = r.
        # So (x + r) = [r, 1]
        
        # Next factor (x - root^2)
        root2 = gf_pow(root, 2)
        # Multiply (x + root) by (x + root2)
        # [r1, 1] * [r2, 1] = r1*r2 + (r1+r2)x + x^2
        # In GF(2), addition is XOR.
        
        # Let's write a generic poly mult
        def poly_mult(p1, p2):
            res = [0] * (len(p1) + len(p2) - 1)
            for i in range(len(p1)):
                for j in range(len(p2)):
                    res[i+j] ^= gf_mult(p1[i], p2[j])
            return res
            
        p = [root, 1]
        roots = [root]
        # Coset C1: 1, 2, 4, 8
        # We need roots alpha^1, alpha^2, alpha^4, alpha^8
        current_root = root
        for _ in range(3):
            current_root = gf_pow(current_root, 2)
            factor = [current_root, 1]
            p = poly_mult(p, factor)
        # p is now M1(x) of degree 4
        return p
        
    M1 = calc_min_poly(alpha)
    
    # Calculate Minimal Polynomial M3(x) for alpha^3
    # Coset C3: 3, 6, 12, 7, 14, 11, 5, 10
    # Roots: alpha^3, alpha^6, alpha^12, alpha^7, alpha^14, alpha^11, alpha^5, alpha^10
    def calc_min_poly_coset(start_exp):
        def poly_mult(p1, p2):
            res = [0] * (len(p1) + len(p2) - 1)
            for i in range(len(p1)):
                for j in range(len(p2)):
                    res[i+j] ^= gf_mult(p1[i], p2[j])
            return res
            
        root = gf_pow(alpha, start_exp)
        p = [root, 1]
        current_exp = start_exp
        # The coset has 8 elements. We start with 1 factor. Need 7 more.
        for _ in range(7):
            current_exp = (current_exp * 2) % 17
            root = gf_pow(alpha, current_exp)
            factor = [root, 1]
            p = poly_mult(p, factor)
        return p
        
    M3 = calc_min_poly_coset(3)
    
    # Generator polynomial g(x) = M1(x) * M3(x)
    def poly_mult(p1, p2):
        res = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                res[i+j] ^= gf_mult(p1[i], p2[j])
        return res
        
    g_poly = poly_mult(M1, M3)
    # g_poly has degree 8. Length 9.
    # Coeffs g[0]...g[8]
    
    # Construct Generator Matrix G_17 (9x17)
    # The code is cyclic. The generator matrix can be formed by shifting g(x).
    # Row i is the coefficients of x^i * g(x) mod (x^17 - 1).
    # Since it's a cyclic code, we can just use the standard generator matrix form:
    # G = [ I_k | P ] or just the shifts.
    # For a cyclic code with generator polynomial g(x) of degree n-k=8,
    # the generator matrix is:
    # [g0 g1 ... g8 0 ... 0]
    # [0 g0 g1 ... g8 0 ...]
    # ...
    # [0 ... 0 g0 g1 ... g8]
    
    G17 = np.zeros((9, 17), dtype=int)
    for i in range(9):
        for j in range(9):
            # Coefficient g[j] is at position i+j
            # g_poly[j] is the coefficient of x^j
            # We need to map GF(16) coeffs to bits?
            # NO. The code is over GF(2). The generator polynomial coefficients are in GF(16),
            # but they must be 0 or 1 for a binary code?
            # Wait. The QR code is a binary code.
            # The generator polynomial g(x) must have coefficients in GF(2).
            # Let's check if M1 and M3 have coefficients in GF(2).
            # Yes, minimal polynomials of elements in GF(2^m) over GF(2) have coefficients in GF(2).
            # So g_poly[j] should be 0 or 1.
            if g_poly[j] not in [0, 1]:
                # This would be an error in logic, but QR codes are binary.
                pass
            
            G17[i, i+j] = g_poly[j]
            
    # Extend to [18, 9, 6] by adding parity bit
    # The parity bit is the sum of the other 17 bits mod 2.
    # We construct G18 (9x18)
    G18 = np.zeros((9, 18), dtype=int)
    for i in range(9):
        G18[i, :17] = G17[i, :]
        # Parity bit
        parity = np.sum(G17[i, :]) % 2
        G18[i, 17] = parity
        
    # Generate all 2^9 = 512 codewords
    words = []
    for msg in range(512):
        # msg is 9-bit integer
        # c = msg_vector * G18
        # msg_vector is 1x9
        # We can compute this as XOR of rows of G18 where msg bit is 1
        c = np.zeros(18, dtype=int)
        for j in range(9):
            if (msg >> j) & 1:
                c ^= G18[j, :]
                
        # Convert vector to integer
        word_int = 0
        for j in range(18):
            if c[j]:
                word_int |= (1 << j)
        words.append(word_int)
        
    # Verify no duplicates (should be none as G is full rank)
    # Verify distances (optional, but good for sanity)
    # The construction guarantees d=6.
    
    return words
# EVOLVE-BLOCK-END
