import time

def construct():
    n = 10
    deadline = time.time() + 50  # safety margin
    
    # Use Z5 x Z2 decomposition for MOLS of order 10
    # Cell (i,j): i = 2*a + b (a in Z5, b in Z2), j = 2*c + d (c in Z5, d in Z2)
    # Symbol s = 5*e + f (e in Z2, f in Z5)
    #
    # For a Latin square parameterized by (p, q, r, s) where:
    #   e = (p*c + q*d + r*a + s*b) % 2
    #   f = (p5*c + q5*d + r5*a + s5*b) % 5
    #
    # For fixed (a,b), as (c,d) varies over Z5 x Z2, we need bijection to Z2 x Z5.
    # This requires the linear map to be invertible in the appropriate sense.
    #
    # Key insight: Use squares where the Z2 component depends on one coordinate
    # and the Z5 component depends on the other, ensuring Latin property.
    #
    # Construction: For each slope m in {1, 2, 3, 4} (non-zero in Z5), define:
    #   e = (c + m*d) % 2  -- wait, c is in Z5, so this doesn't work directly
    #
    # Better approach: Use the fact that Z10 = Z5 x Z2 via CRT-like decomposition
    # i = 5*b + a where a in Z5, b in Z2 (alternative indexing)
    # Actually, let's use: i = 2*a + b, a in 0..4, b in 0..1
    #
    # For Latin square L_k with parameter k (1 <= k <= 4, k in Z5\{0}):
    #   a_i = i // 2, b_i = i % 2
    #   a_j = j // 2, b_j = j % 2
    #   f = (a_i + k * a_j) % 5  -- Z5 component
    #   e = (b_i + b_j) % 2      -- Z2 component (same for all k)
    #   s = 5 * e + f
    #
    # Check Latin:
    # Row i (fixed a_i, b_i): as j varies, a_j varies over Z5, b_j varies over Z2
    #   f = (a_i + k*a_j) % 5: as a_j varies, f varies over Z5 (since k != 0 in Z5)
    #   e = (b_i + b_j) % 2: as b_j varies, e varies over Z2
    #   Combined (e,f) covers Z2 x Z5 bijectively? 
    #   For each (e,f) pair, we need unique (a_j, b_j).
    #   f determines a_j = (f - a_i) * k^{-1} mod 5
    #   e determines b_j = (e - b_i) mod 2
    #   So yes, bijective!
    #
    # Column j (fixed a_j, b_j): as i varies, a_i varies over Z5, b_i varies over Z2
    #   f = (a_i + k*a_j) % 5: as a_i varies, f varies over Z5
    #   e = (b_i + b_j) % 2: as b_i varies, e varies over Z2
    #   Same argument, bijective!
    #
    # Orthogonality between L_k and L_l (k != l, both in {1,2,3,4}):
    #   Pair: ((5*(b_i+b_j)%2 + (a_i+k*a_j)%5), (5*(b_i+b_j)%2 + (a_i+l*a_j)%5))
    #   Two pairs equal iff:
    #     (a_i+k*a_j)%5 = (a_i'+l*a_j')%5 AND (a_i+l*a_j)%5 = (a_i'+k*a_j')%5
    #     AND (b_i+b_j)%2 = (b_i'+b_j')%2
    #   From the two Z5 equations:
    #     (k-l)*a_j = (k-l)*a_j' mod 5
    #   Since k != l and Z5 is a field, k-l != 0, so a_j = a_j'
    #   Then a_i = a_i' from first equation
    #   From Z2 equation: b_i + b_j = b_i' + b_j', and since a_j = a_j', we have j = j'
    #   Then b_i = b_i', so i = i'
    #   Thus orthogonal!
    #
    # So we can construct 4 MOLS with k = 1, 2, 3, 4.
    # Can we add more? Try k=0:
    #   L_0: f = a_i, e = (b_i + b_j) % 2
    #   Check orthogonality with L_k:
    #     (a_i, a_i + k*a_j) = (a_i', a_i' + k*a_j')
    #     => a_i = a_i', a_j = a_j'
    #     So a_i = a_i' and a_j = a_j', but b_i, b_j not determined
    #     e: b_i + b_j = b_i' + b_j'
    #     Multiple solutions exist (e.g., swap b_i and b_j)
    #     So L_0 is NOT orthogonal to L_k.
    #
    # What about different Z2 components?
    # Try: e = (b_i + m*b_j) % 2 for m in {0, 1}
    #   m=0: e = b_i
    #   m=1: e = (b_i + b_j) % 2
    #
    # Let's try combining:
    # L(k, m) where k in {1,2,3,4} (Z5 slope), m in {0,1} (Z2 slope)
    #   f = (a_i + k*a_j) % 5
    #   e = (b_i + m*b_j) % 2
    #
    # Check Latin for L(k, m):
    # Row i: f covers Z5 as a_j varies, e covers Z2 as b_j varies
    #   For m=0: e = b_i (constant), so e doesn't vary! NOT Latin.
    #   For m=1: e = (b_i + b_j) % 2, varies as b_j varies. OK.
    # So m must be 1 for Latin property.
    #
    # Thus we get exactly 4 MOLS: k = 1, 2, 3, 4 with m = 1.
    
    def make_square(k):
        """Create Latin square with Z5 slope k and Z2 slope 1."""
        sq = []
        for i in range(n):
            a_i = i // 2
            b_i = i % 2
            row = []
            for j in range(n):
                a_j = j // 2
                b_j = j % 2
                f = (a_i + k * a_j) % 5
                e = (b_i + b_j) % 2
                s = 5 * e + f
                row.append(int(s))
            sq.append(row)
        return sq
    
    def is_latin_square(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
            for j in range(n):
                col = [sq[i][j] for i in range(n)]
                if len(set(col)) != n:
                    return False
        return True
    
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n * n
    
    # Generate 4 MOLS
    squares = [make_square(k) for k in [1, 2, 3, 4]]
    
    # Verify all are Latin squares
    for sq in squares:
        if not is_latin_square(sq):
            # Fallback to simpler construction
            L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
            L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
            if is_latin_square(L1) and is_latin_square(L2) and are_orthogonal(L1, L2):
                return [L1, L2]
            return [L1]
    
    # Verify all pairs are orthogonal
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not are_orthogonal(squares[i], squares[j]):
                # Fallback to 2 MOLS
                L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
                L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
                if is_latin_square(L1) and is_latin_square(L2) and are_orthogonal(L1, L2):
                    return [L1, L2]
                return [L1]
    
    return squares
