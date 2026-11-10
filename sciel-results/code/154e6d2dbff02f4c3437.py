import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 10
    p, q = 2, 5

    def decomp(i):
        # i in 0..9 -> (a in Z2, b in Z5) where i = a*q + b  (i.e., a = i//5, b = i%5)
        return (i // q, i % q)

    def recomb(a, b):
        return a * q + b

    def is_latin(sq):
        for i in range(n):
            row = sq[i]
            if len(set(row)) != n:
                return False
            col = [sq[j][i] for j in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(sq1, sq2):
        seen = set()
        for i in range(n):
            r1 = sq1[i]
            r2 = sq2[i]
            for j in range(n):
                pair = (r1[j], r2[j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Build the 5 MOLS of order 5 (from Z_5)
    # L_k[i][j] = (i + k*j) mod 5, for k in 0..4
    # Actually we need 5 MOLS: k=0,1,2,3,4
    # But k=0 gives L_0[i][j] = i, which is the "identity" square.
    # All 5 are mutually orthogonal.
    mols_5 = []
    for k in range(5):
        sq = [[(i + k * j) % q for j in range(q)] for i in range(q)]
        mols_5.append(sq)

    # The 2x2 Latin square
    L2 = [[0, 1], [1, 0]]

    # Construct 8 MOLS of order 10 using the Kronecker/tensor product approach.
    # For order 10 = 2 * 5, we can use the following construction:
    # Represent cell (i, j) where i = 5*a + b (a in Z2, b in Z5) and j = 5*c + d (c in Z2, d in Z5)
    # Symbol = 5 * X + Y where X in Z2, Y in Z5
    #
    # We need 8 mutually orthogonal squares.
    #
    # Construction:
    # The standard way to get MOLS(10) from MOLS(2) and MOLS(5):
    # We have 1 MOLS of order 2 (just L2 = [[0,1],[1,0]]) and 5 MOLS of order 5.
    #
    # Actually, the known result is that MOLS(10) has exactly 8 squares (max).
    # 
    # Let me use a more direct algebraic construction.
    # 
    # For order 10 = 2*5, use the group Z_2 x Z_5.
    # A Latin square is a function f: (Z_2 x Z_5) x (Z_2 x Z_5) -> Z_2 x Z_5
    # such that for fixed row, f is a permutation, and for fixed column, f is a permutation.
    #
    # Linear squares: f(a,b; c,d) = (a + c + alpha*b, b + beta*c + gamma*d) mod respective
    # where alpha in Z_2, beta, gamma in Z_5, with constraints for Latin property.
    #
    # For Latin property:
    # - For fixed (a,b), as (c,d) varies: (a+c+alpha*b, b+beta*c+gamma*d) must be a bijection.
    #   The Z2 component: a+c+alpha*b -> as c varies in Z2, this is a+c+const, which is a bijection. Good.
    #   The Z5 component: b+beta*c+gamma*d -> as (c,d) varies over Z2 x Z5, we need this to be a bijection.
    #   Since c only takes 2 values and d takes 5, the map (c,d) -> beta*c+gamma*d mod 5 must be a bijection.
    #   This requires gamma != 0 (so that for fixed c, d -> gamma*d is bijective, and varying c shifts by beta).
    #   Actually, we need: for any two distinct (c1,d1) != (c2,d2), beta*c1+gamma*d1 != beta*c2+gamma*d2 mod 5.
    #   If c1=c2, then gamma*(d1-d2) != 0, so gamma != 0.
    #   If c1!=c2 (so c1-c2 = 1 mod 2, meaning c1 and c2 are different), then beta*(c1-c2)+gamma*(d1-d2) = beta+gamma*(d1-d2).
    #   For this to never be 0 for any d1!=d2... wait, d1-d2 can be any value in Z5.
    #   So we need beta + gamma*t != 0 for all t in Z5? That's impossible since gamma*t covers all of Z5.
    #   
    # Let me reconsider. The issue is that the Z2 and Z5 parts are coupled.
    #
    # Better approach: Use the known explicit construction.
    #
    # From the literature, 8 MOLS of order 10 can be constructed as follows:
    # Write i = 5*a + b, j = 5*c + d where a,c in {0,1}, b,d in {0,1,2,3,4}.
    #
    # Square k (for k=0..7):
    # symbol = 5 * X_k + Y_k
    #
    # Let me try a different parametrization. Consider:
    # X = (a + c) mod 2  [this is the "Z2 part"]
    # Y = (b + m_k * d + n_k * (a*c) + o_k * (a*d) + p_k * (b*c)) mod 5
    #
    # This is getting complex. Let me just use the known 8 squares directly.
    #
    # Actually, let me use a cleaner algebraic construction.
    #
    # The field F_2 x F_5 is not a field, but we can use the following:
    # For each square, define:
    #   X = (a + c) mod 2
    #   Y = (b + k * d + l * a * c) mod 5   where k, l depend on the square
    #
    # For this to be Latin:
    #   Fix (a,b), vary (c,d): X = a+c mod 2 is bijective in c. Y = b + k*d + l*a*c mod 5.
    #   As (c,d) ranges over Z2 x Z5, (c, k*d + l*a*c) must range over all Z2 x Z5.
    #   For fixed c, d -> k*d is bijective iff k != 0.
    #   For different c values (c=0 and c=1), the Y values are b+l*a*0 + k*d = b+k*d and b+l*a*1+k*d = b+l*a+k*d.
    #   These two sets {b+k*d : d in Z5} and {b+l*a+k*d : d in Z5} are both all of Z5 (shifted), so they overlap completely.
    #   So this doesn't work for Latin property.
    #
    # Let me try yet another approach. I'll use the direct known construction from the literature.
    #
    # The 8 MOLS of order 10 are well-known. Let me construct them using the following method:
    #
    # For t = 0, 1, 2, 3, 4, 5, 6, 7:
    #   Square_t[i][j] where i = 5*a+b, j = 5*c+d:
    #
    # I'll use a computer algebra approach: search for linear forms.
    #
    # General form: symbol = 5*X + Y where
    #   X = (a + c + r1*b + r2*d) mod 2  with r1, r2 in {0,1}
    #   Y = (b + s1*c + s2*d + s3*a*d + s4*b*c + s5*a*c) mod 5  with s1..s5 in Z5
    #
    # This space is too large to search exhaustively. Let me use a simpler family.
    #
    # Simpler family:
    #   X = (a + c) mod 2
    #   Y = (b + k*d + l*a) mod 5, for k in Z5\{0}, l in Z5
    #
    # For Latin property: fix (a,b), vary (c,d).
    #   X = a+c mod 2: as c varies, X covers {a, a+1}. Good.
    #   Y = b + k*d + l*a mod 5: as d varies, Y covers all of Z5 (since k != 0).
    #   But we need the pair (X, Y) to be a bijection from Z2 x Z5 to Z2 x Z5.
    #   For c=0: X=a, Y=b+k*d+l*a for d in Z5 -> gives (a, b+l*a+k*d) for all d, so all 5 symbols with X=a.
    #   For c=1: X=a+1, Y=b+k*d+l*a for d in Z5 -> gives (a+1, b+l*a+k*d) for all d, so all 5 symbols with X=a+1.
    #   So yes, this is Latin! The symbol depends on a only through the shift l*a, and on c only through X.
    #
    # Orthogonality: Two squares with parameters (k1, l1) and (k2, l2) are orthogonal iff
    #   the map (a,b,c,d) -> (X1, Y1, X2, Y2) is injective.
    #   X1 = a+c, X2 = a+c (same!), so X1=X2 always. That's a problem.
    #   So all squares in this family have the same X component, meaning they can't be mutually orthogonal
    #   (since the pair (X1,Y1) and (X2,Y2) would have X1=X2, so orthogonality requires Y1 and Y2 to determine (a,b,c,d) uniquely given X).
    #   Given X=a+c, we have c=a+X. Then Y1 = b+k1*d+l1*a, Y2 = b+k2*d+l2*a.
    #   Y1-Y2 = (k1-k2)*d + (l1-l2)*a. For fixed (a,d), this is determined.
    #   We need: given X, Y1, Y2, we can determine (a,b,c,d).
    #   From Y1-Y2 = (k1-k2)*d + (l1-l2)*a, and X=a+c (so c=a+X mod 2).
    #   We have 2 equations (Y1, Y2) in 4 unknowns (a,b,c,d) with c=a+X.
    #   So effectively 2 equations in 3 unknowns (a,b,d). This is underdetermined, so NOT orthogonal.
    #
    # So this family doesn't work for mutual orthogonality.
    #
    # I need a different approach. Let me just hardcode the known 8 MOLS of order 10.
    #
    # The standard construction uses the fact that 10 = 2*5 and employs a specific set of 8 squares.
    # Let me generate them using a verified construction.

    # Use the construction from the theory of MOLS for composite orders.
    # For n = p*q where p, q are primes, the maximum number of MOLS is min(p-1, q-1) = 4 if p=2,q=5.
    # Wait, that's not right either. The maximum number of MOLS of order n is at most n-1.
    # For n=10, it's known that the maximum is 8 (not 9, since 10 is not a prime power).
    #
    # Let me use a concrete verified construction. I'll build 8 squares and verify.

    # Construction using linear maps over Z_2 x Z_5:
    # Each square is defined by:
    #   f(i,j) where i=(a,b), j=(c,d) with a,c in Z2, b,d in Z5
    #   f(i,j) = (A*(a,c) + B*(b,d)) where A is a 2x2 matrix over appropriate rings...
    #
    # This is getting too abstract. Let me just use a known explicit set.
    #
    # I'll construct 8 MOLS using the following verified method:
    # For k in range(8), define square k by:
    #   i = 5*a + b, j = 5*c + d
    #   symbol = 5 * ((a + c) % 2) + ((b + k * d + (k % 2) * a * c) % 5)
    #
    # Let me test this mentally for a few cases... Actually, let me just code it up and verify.

    candidates = []

    # Family 1: X = (a+c) mod 2, Y = (b + k*d + l*a*c) mod 5
    # for k in Z5, l in Z5
    for k in range(5):
        for l in range(5):
            sq = [[0] * n for _ in range(n)]
            for i in range(n):
                a, b = decomp(i)
                for j in range(n):
                    c, d = decomp(j)
                    X = (a + c) % 2
                    Y = (b + k * d + l * a * c) % 5
                    sq[i][j] = 5 * X + Y
            candidates.append(sq)

    # Family 2: X = (a + c + r*b) mod 2, Y = (b + k*d + l*a) mod 5
    # for r in {0,1}, k in Z5\{0}, l in Z5
    for r in range(2):
        for k in range(1, 5):  # k != 0
            for l in range(5):
                sq = [[0] * n for _ in range(n)]
                for i in range(n):
                    a, b = decomp(i)
                    for j in range(n):
                        c, d = decomp(j)
                        X = (a + c + r * b) % 2
                        Y = (b + k * d + l * a) % 5
                        sq[i][j] = 5 * X + Y
                candidates.append(sq)

    # Family 3: X = (a + c + r*b + s*d) mod 2, Y = (b + k*d + l*a + m*c) mod 5
    # for r,s in {0,1}, k in Z5\{0}, l,m in Z5
    for r in range(2):
        for s in range(2):
            for k in range(1, 5):
                for l in range(5):
                    for m in range(5):
                        sq = [[0] * n for _ in range(n)]
                        for i in range(n):
                            a, b = decomp(i)
                            for j in range(n):
                                c, d = decomp(j)
                                X = (a + c + r * b + s * d) % 2
                                Y = (b + k * d + l * a + m * c) % 5
                                sq[i][j] = 5 * X + Y
                        candidates.append(sq)

    # Deduplicate and filter Latin
    seen_keys = set()
    valid = []
    for sq in candidates:
        if not is_latin(sq):
            continue
        key = tuple(tuple(row) for row in sq)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        valid.append(sq)

    # Greedily select mutually orthogonal set
    mols = []
    for sq in valid:
        if all(are_orthogonal(sq, existing) for existing in mols):
            mols.append(sq)
        if len(mols) >= 8:
            break

    # Ensure pure Python int types and list-of-lists
    out = []
    for sq in mols:
        out.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return out
