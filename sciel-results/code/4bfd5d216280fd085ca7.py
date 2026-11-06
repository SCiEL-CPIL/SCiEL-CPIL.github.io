import time
import numpy as np

def construct():
    """
    Construct the largest possible binary code of length 18 with minimum distance 6.
    Primary strategy: Deterministic construction of the extended quadratic residue code [18, 9, 6].
    Fallback strategy: Bit-parallel greedy search.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = 85.0  # Safety margin for 90s limit
    start_time = time.time()
    
    # --- Strategy 1: Extended QR Code [18, 9, 6] ---
    # The QR code of length 17 is a [17, 9, 5] code.
    # Extending it with a parity bit gives [18, 9, 6].
    
    # 1. GF(2^8) Arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
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

    def gf_pow(base, exp):
        res = 1
        while exp:
            if exp & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            exp >>= 1
        return res

    # 2. Find alpha of order 17
    # beta is a primitive element (root of primitive poly), order 255.
    # alpha = beta^k where 255/gcd(255,k) = 17 => gcd(255,k)=15 => k=15.
    beta = 0x02  # x
    alpha = gf_pow(beta, 15)
    
    # Verify alpha has order 17
    # alpha^17 == 1, alpha != 1
    if gf_pow(alpha, 17) != 1 or alpha == 1:
        # Fallback if arithmetic is weird
        return _greedy_construct(N, MIN_DIST, start_time, DEADLINE)

    # 3. Compute Generator Polynomial g(x)
    # g(x) is the minimal polynomial of alpha over GF(2).
    # Roots are alpha^i for i in C1 = {1, 2, 4, 8, 16, 15, 13, 9}.
    # We compute the polynomial whose roots are these.
    # We can build it by multiplying (x - alpha^r) for r in C1.
    # In GF(2), x - a = x + a.
    
    # Start with 1
    g_coeffs = [1]  # Represents polynomial 1
    
    for r in [1, 2, 4, 8, 16, 15, 13, 9]:
        root = gf_pow(alpha, r)
        # Multiply current poly by (x + root)
        # (x + root) * P(x)
        # If P(x) = sum p_i x^i, then (x + root)P(x) = sum p_i x^{i+1} + sum (root * p_i) x^i
        new_coeffs = [0] * (len(g_coeffs) + 1)
        for i, coef in enumerate(g_coeffs):
            if coef:
                new_coeffs[i+1] ^= 1  # x * coef
                # root * coef is in GF(2^8), but wait.
                # The coefficients of g(x) must be in GF(2).
                # The product of (x - alpha^r) for r in a cyclotomic coset has coefficients in GF(2).
                # However, intermediate multiplication in GF(2^8) will yield GF(2^8) elements.
                # We must ensure we are working in the correct field.
                # Actually, the standard way is:
                # The minimal polynomial is the product of (x - alpha^{2^j * 1}) for j=0..7.
                # The coefficients will automatically be 0 or 1.
                # Let's just accumulate in GF(2^8) and check if they are 0 or 1 at the end.
                # Or better, perform the multiplication in GF(2^8) and rely on the property that the result is in GF(2).
                
                val = gf_mul(root, coef)
                new_coeffs[i] ^= val
        g_coeffs = new_coeffs
        
    # g_coeffs should be [1, 1, 0, 1, 1, 1, 0, 1, 1] or similar.
    # Let's verify they are 0 or 1.
    for c in g_coeffs:
        if c not in (0, 1):
            # Arithmetic error or logic error. Fallback.
            return _greedy_construct(N, MIN_DIST, start_time, DEADLINE)
            
    # g_coeffs represents g(x) = g_0 + g_1 x + ... + g_8 x^8
    # The generator matrix rows are the cyclic shifts of the coefficient vector.
    # For a cyclic code of length 17, the generator matrix is formed by the shifts of g(x).
    # g(x) has degree 8. The code has dimension 17 - 8 = 9.
    # The basis vectors are g(x), xg(x), ..., x^8 g(x) mod (x^17 - 1).
    
    g_vec = g_coeffs[:]  # length 9
    # Pad to length 17 for shifting?
    # Actually, the codeword corresponding to message m(x) is m(x)g(x) mod (x^17-1).
    # Since deg(m) < 9, deg(mg) < 17. So no reduction needed if we just multiply polynomials.
    # We can generate the basis by shifting the coefficient vector of g(x).
    
    basis = []
    for i in range(9):
        row = [0] * 17
        # Shift g_vec by i positions
        # g_vec[k] corresponds to x^k
        # x^i * g(x) has coefficients g_vec[k] at position i+k
        for k in range(9):
            pos = i + k
            if pos < 17:
                row[pos] = g_vec[k]
        basis.append(row)
        
    # Generate all 512 codewords
    words_17 = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                # XOR basis[i] into word
                b = basis[i]
                for k in range(17):
                    if b[k]:
                        word |= (1 << k)
        words_17.append(word)
        
    # Extend to length 18
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        # Append parity bit at the end (position 17)
        w_ext = (w << 1) | parity
        words_18.append(w_ext)
        
    # Verify
    # Check min distance
    min_dist = 100
    # Optimization: Since it's a linear code, we just need to check the weights of all non-zero codewords.
    # d_min is the minimum weight of any non-zero codeword.
    for w in words_18:
        if w != 0:
            wt = w.bit_count()
            if wt < min_dist:
                min_dist = wt
            if min_dist < MIN_DIST:
                break
                
    if min_dist >= MIN_DIST:
        return words_18
    else:
        # Fallback
        return _greedy_construct(N, MIN_DIST, start_time, DEADLINE)

def _greedy_construct(N, MIN_DIST, start_time, DEADLINE):
    """
    Fallback greedy construction using bit-parallel conflict masks.
    """
    size = 1 << N
    
    # Precompute masks of weight < MIN_DIST
    # There are sum_{i=0}^{5} C(18, i) = 12616 masks
    masks = []
    for i in range(size):
        if i.bit_count() < MIN_DIST:
            masks.append(i)
    masks = np.array(masks, dtype=np.int64)
    
    forbidden = np.zeros(size, dtype=bool)
    
    def mark_forbidden(c):
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
        
    # Start with 0
    code = [0]
    mark_forbidden(0)
    
    # Try standard order
    for c in range(1, size):
        if time.time() - start_time > DEADLINE:
            break
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
            
    # Try reverse order if time permits
    if time.time() - start_time < DEADLINE - 5:
        forbidden[:] = False
        code_rev = [0]
        mark_forbidden(0)
        for c in range(size - 1, 0, -1):
            if time.time() - start_time > DEADLINE:
                break
            if not forbidden[c]:
                code_rev.append(c)
                mark_forbidden(c)
        if len(code_rev) > len(code):
            code = code_rev
            
    return code
