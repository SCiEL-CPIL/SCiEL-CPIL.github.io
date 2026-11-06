    def construct():
        import time
        start_time = time.time()
        
        # 1. Generate [17,8,6] BCH code
        # Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1
        p_int = 0b100011101
        
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                while a >= 256:
                    a ^= p_int
            return res
            
        def gf_pow(base, exp):
            res = 1
            while exp:
                if exp & 1:
                    res = gf_mul(res, base)
                base = gf_mul(base, base)
                exp >>= 1
            return res
        
        # Find primitive 17th root alpha
        # alpha = beta^15 where beta is root of p_int
        # We need beta. Let's just pick an element and check order.
        # Actually, we can just iterate through elements to find one of order 17.
        alpha = None
        for i in range(1, 256):
            # Check if order is 17
            # i^17 == 1 and i^1 != 1
            # This is slow.
            pass
        
        # Better: Use the known fact that beta^15 is a primitive 17th root if beta is primitive.
        # Is beta (root of x^8+x^4+x^3+x^2+1) primitive?
        # Yes, it is a primitive polynomial.
        # So alpha = beta^15.
        # We need to represent beta.
        # beta is the class of x in GF(2^8)/p(x).
        # So beta = 2 (binary 10).
        beta = 2
        alpha = gf_pow(beta, 15)
        
        # Verify alpha^17 == 1
        # assert gf_pow(alpha, 17) == 1
        
        def find_min_poly(alpha_pow):
            # Find min poly of alpha_pow
            # Sequence 1, a, a^2, ...
            # We look for c0 + c1*a + ... + ck*a^k = 0
            # We can do this by finding the linear dependence of vectors in GF(2)^8
            # But the coefficients are in GF(2), not GF(2^8).
            # The minimal polynomial is over GF(2).
            # So we just need to find k and coefficients c_i in {0,1}.
            
            # We maintain a list of powers: v[0]=1, v[1]=a, v[2]=a^2 ...
            # And we want to find if v[k] is in span(v[0]...v[k-1]) over GF(2)?
            # No, the span is over GF(2) of the elements in GF(2^8).
            # This is equivalent to finding the characteristic polynomial of the linear map L(x) = a*x.
            
            # Simpler:
            # m(x) = x.
            # r = alpha_pow.
            # We want m(alpha_pow) = 0.
            # Let m(x) = x^k + c_{k-1}x^{k-1} + ... + c_0.
            # This means alpha_pow^k + c_{k-1}alpha_pow^{k-1} + ... + c_0 = 0.
            
            # We can iterate k from 1 to 8 (max degree).
            # For each k, we have a system of 8 linear equations (one for each bit of the result)
            # with k unknowns c_0...c_{k-1}.
            # We want a non-trivial solution? No, we want the specific relation.
            # Actually, for a given k, there is at most one solution for c if the powers are linearly independent?
            
            # Let's just try k=1..8.
            # For a fixed k, we want to find c_0..c_{k-1} such that:
            # sum_{i=0}^{k-1} c_i * alpha_pow^i = alpha_pow^k
            # This is a linear system over GF(2) with 8 equations and k unknowns.
            # We can solve it using Gaussian elimination.
            
            def solve_gf2(A, b):
                # A is k x 8 matrix (list of lists of ints 0/1)
                # b is 8x1 vector
                # We want x such that Ax = b
                # A has rows a_0, ..., a_{k-1}
                # x has entries c_0, ..., c_{k-1}
                # Equation: sum_j A[i][j] x[j] = b[i]
                # This is standard Gaussian elimination.
                n = 8
                m = len(A)
                # Augmented matrix
                M = [row[:] + [b[i]] for i, row in enumerate(A)]
                
                rank = 0
                for col in range(m):
                    # Find pivot
                    pivot = -1
                    for row in range(rank, n):
                        if M[row][col]:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    M[rank], M[pivot] = M[pivot], M[rank]
                    for row in range(n):
                        if row != rank and M[row][col]:
                            for c in range(m + 1):
                                M[row][c] ^= M[rank][c]
                    rank += 1
                    if rank == m:
                        break
                
                # Check consistency
                for row in range(rank, n):
                    if M[row][m]:
                        return None # No solution
                
                # Solution is in the last column for the pivot columns
                # But we need the specific solution where non-pivot variables are 0?
                # Since we are looking for the minimal polynomial, the solution is unique if it exists.
                x = [0] * m
                for i in range(rank):
                    # Find the pivot column for row i
                    # This logic is getting complicated.
                    pass
                return None # Placeholder
            
            # Actually, there is a simpler way.
            # Just compute the powers and check for linear dependence over GF(2) of the *vectors*?
            # No, the coefficients are in GF(2).
            
            # Let's use the property:
            # The minimal polynomial is the characteristic polynomial of the companion matrix?
            
            # Let's just brute force k=1..8.
            # For each k, we have the equation:
            # c_0 * 1 + c_1 * a + ... + c_{k-1} * a^{k-1} = a^k
            # This is a system of 8 linear equations over GF(2) with k unknowns.
            # We can solve it.
            
            for k in range(1, 9):
                # Build matrix A (8xk) and vector b (8x1)
                # A[i][j] is the i-th bit of a^j
                # b[i] is the i-th bit of a^k
                
                a_powers = [1]
                for j in range(1, k):
                    a_powers.append(gf_mul(a_powers[-1], alpha_pow))
                target = gf_mul(a_powers[-1], alpha_pow) if k > 1 else alpha_pow
                
                # Matrix A: rows are bits of a^j
                # We want sum c_j * a^j = target
                # So for each bit i: sum_j c_j * (a^j)_i = target_i
                
                # We can solve this using Gaussian elimination on the transpose?
                # Or just iterate through all 2^k possibilities if k is small?
                # k is at most 8. 2^8 = 256.
                # This is very fast!
                
                for mask in range(1 << k):
                    res = 0
                    for j in range(k):
                        if (mask >> j) & 1:
                            res ^= a_powers[j]
                    if res == target:
                        # Found coefficients
                        # Polynomial is x^k + sum c_j x^j
                        # We need to return the polynomial coefficients
                        # c_k = 1
                        # c_j = (mask >> j) & 1
                        return [1] + [((mask >> j) & 1) for j in range(k)]
            return None
        
        # Find min polys for alpha^1, alpha^3, alpha^5
        m1 = find_min_poly(alpha)
        m3 = find_min_poly(gf_pow(alpha, 3))
        m5 = find_min_poly(gf_pow(alpha, 5))
        
        # LCM of polynomials over GF(2)
        # Since they are minimal polynomials of conjugates, they are distinct and coprime?
        # If they are distinct, LCM is product.
        def poly_mul(p, q):
            # p, q are lists of coeffs, p[0] is constant term?
            # My find_min_poly returns [1, c0, c1...]?
            # I returned [1] + [c0...]. So p[0]=1 (coeff of x^k), p[k]=c0.
            # Let's standardize: list of coeffs from highest degree to lowest.
            # So p[0] is coeff of x^deg, p[deg] is constant.
            res = [0] * (len(p) + len(q) - 1)
            for i in range(len(p)):
                for j in range(len(q)):
                    if p[i] and q[j]:
                        res[i+j] ^= 1
            # Remove leading zeros
            while len(res) > 1 and res[0] == 0:
                res.pop(0)
            return res
        
        # Adjust find_min_poly output format
        # It returned [1, c0, c1, ..., c_{k-1}] where 1 is coeff of x^k.
        # This is already high-to-low.
        
        g_poly = m1
        g_poly = poly_mul(g_poly, m3)
        g_poly = poly_mul(g_poly, m5)
        
        # g_poly should have length 10 (degree 9)
        # Generate codewords
        # Codeword is m(x)g(x) mod (x^17-1)
        # m(x) is 8 bits.
        
        words = []
        for msg in range(256):
            # m(x) coeffs
            m_poly = [((msg >> i) & 1) for i in range(8)] # low to high
            # Reverse to high to low?
            # Let's just do polynomial multiplication.
            
            # g_poly is high to low.
            # Let's convert m_poly to high to low.
            m_high = m_poly[::-1]
            
            # Multiply
            prod = [0] * (8 + len(g_poly) - 1)
            for i in range(8):
                for j in range(len(g_poly)):
                    if m_high[i] and g_poly[j]:
                        prod[i+j] ^= 1
            
            # Reduce mod x^17 - 1
            # prod has length up to 8+9-1 = 16.
            # Wait, degree of m is 7, degree of g is 9. Degree of prod is 16.
            # So length is 17.
            # No reduction needed if length <= 17.
            # If length > 17, we need to reduce.
            # But 17 is the length.
            
            # Convert to int
            w = 0
            for i in range(len(prod)):
                if prod[i]:
                    # prod[0] is x^16 coeff.
                    # We want bit 16 to be prod[0]?
                    # Standard: bit 0 is LSB.
                    # If prod[0] is coeff of x^16, it should be bit 16.
                    w |= 1 << (len(prod) - 1 - i)
            
            words.append(w)
            
        # Greedy add
        code_set = set(words)
        
        # Deadline check
        if time.time() - start_time > 40:
            return list(code_set)
            
        # Try to add more
        # Iterate all 17-bit integers
        for x in range(1 << 17):
            if x in code_set:
                continue
            # Check distance
            valid = True
            for c in code_set:
                if ((x ^ c).bit_count() < 6):
                    valid = False
                    break
            if valid:
                code_set.add(x)
                
        if time.time() - start_time > 50:
            return list(code_set)
            
        # Second pass with random order?
        # If we didn't reach 258, try random shuffling
        if len(code_set) < 258:
            import random
            candidates = [x for x in range(1 << 17) if x not in code_set]
            random.shuffle(candidates)
            for x in candidates:
                valid = True
                for c in code_set:
                    if ((x ^ c).bit_count() < 6):
                        valid = False
                        break
                if valid:
                    code_set.add(x)
                if time.time() - start_time > 55:
                    break
            
        return list(code_set)
    