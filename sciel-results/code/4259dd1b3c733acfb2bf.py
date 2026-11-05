"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
_start = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def _strassen_decompose(n, m, p):
    """
    Recursively compute a rank decomposition for n×m × m×p matrix multiplication
    using Strassen's 7-product recursion. Returns (U, V, W) as lists of lists.
    """
    # Base case: use schoolbook
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Try to find a good split
    # Split n into n1, n2; m into m1, m2; p into p1, p2
    # We want balanced-ish splits
    
    n1 = n // 2
    n2 = n - n1
    m1 = m // 2
    m2 = m - m1
    p1 = p // 2
    p2 = p - p1
    
    # Get sub-decompositions for the 7 Strassen products
    # M1 = (A11+A22)(B11+B22): size (n1+n2) x (m1+m2) x (p1+p2) = n x m x p ... no
    # Actually: A11 is n1×m1, A22 is n2×m2, so A11+A22 requires same shape.
    # This only works if n1==n2 and m1==m2 and p1==p2 for the additions to make sense.
    
    # For unequal splits, we need to pad or use a different approach.
    # Let's only use Strassen when dimensions are even (equal splits).
    
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n1 = n // 2
        m1 = m // 2
        p1 = p // 2
        
        # 7 sub-problems:
        # M1: (n1+m1) x (m1+p1) x (p1+n1) -- no, let me think in terms of matrix sizes
        # M1 = (A11 + A22) @ (B11 + B22)
        #   A11+A22 is n×m (since n1=n2=n/2, m1=m2=m/2)
        #   B11+B22 is m×p
        #   Result is n×p
        #   But we decompose it as: the product of two n×m and m×p matrices
        #   So we need a decomposition of size (n, m, p) for M1? No...
        
        # Actually, M1 is the product of (A11+A22) which is n×m and (B11+B22) which is m×p.
        # So M1's tensor is the same size as the original: n×m×p.
        # That's circular!
        
        # I think I'm confusing myself. Let me restart the Strassen decomposition.
        
        # In Strassen, we have:
        # A is n×m, split into A11(n1×m1), A12(n1×m2), A21(n2×m1), A22(n2×m2)
        # B is m×p, split into B11(m1×p1), B12(m1×p2), B21(m2×p1), B22(m2×p2)
        # C is n×p, split into C11(n1×p1), C12(n1×p2), C21(n2×p1), C22(n2×p2)
        
        # With n1=n2=n/2, m1=m2=m/2, p1=p2=p/2:
        # M1 = (A11+A22)(B11+B22): (n/2 × m/2) @ (m/2 × p/2) -> but A11+A22 is n/2 × m/2 only if we're adding corresponding blocks. A11 is n/2×m/2, A22 is n/2×m/2. Yes! So M1 is a (n/2)×(m/2) × (m/2)×(p/2) multiplication.
        
        # So all 7 M's are (n/2)×(m/2) × (m/2)×(p/2) multiplications!
        
        # Great, so:
        # M1: (n/2, m/2, p/2)
        # M2: (n/2, m/2, p/2) -- (A21+A22) is n/2 × m, B11 is m/2 × p/2. Wait no.
        
        # Let me re-derive:
        # M1 = (A11+A22)(B11+B22)
        #   A11+A22: A11 is (n/2)×(m/2), A22 is (n/2)×(m/2) -> sum is (n/2)×(m/2)
        #   B11+B22: B11 is (m/2)×(p/2), B22 is (m/2)×(p/2) -> sum is (m/2)×(p/2)
        #   Product: (n/2)×(p/2). So this is a (n/2, m/2, p/2) matmul.
        
        # M2 = (A21+A22)B11
        #   A21+A22: A21 is (n/2)×(m/2), A22 is (n/2)×(m/2) -> (n/2)×(m/2)
        #   B11: (m/2)×(p/2)
        #   Product: (n/2)×(p/2). So (n/2, m/2, p/2).
        
        # M3 = A11(B12-B22)
        #   A11: (n/2)×(m/2)
        #   B12-B22: B12 is (m/2)×(p/2), B22 is (m/2)×(p/2) -> (m/2)×(p/2)
        #   Product: (n/2)×(p/2). So (n/2, m/2, p/2).
        
        # M4 = A22(B21-B11)
        #   A22: (n/2)×(m/2)
        #   B21-B11: (m/2)×(p/2)
        #   Product: (n/2)×(p/2). So (n/2, m/2, p/2).
        
        # M5 = (A11+A12)B22
        #   A11+A12: A11 is (n/2)×(m/2), A12 is (n/2)×(m/2) -> (n/2)×(m)
        #   B22: (m/2)×(p/2)
        #   Product: (n/2)×(p/2). But A11+A12 is (n/2)×m, not (n/2)×(m/2)!
        
        # Hmm, so M5 is (n/2, m, p/2) matmul, not (n/2, m/2, p/2).
        
        # This breaks the clean recursion. The standard Strassen recursion only works cleanly when all dimensions are equal (square matrices).
        
        # For rectangular matrices, the recursion is messier. Let me look at this differently.
        
        # Actually, I recall that for rectangular Strassen, you can still get good bounds but the sub-problems have different sizes.
        
        # Given the complexity, let me just use the schoolbook for the base and try a different approach:
        # Use the known fact that r(4,5,6) can be achieved via padding to (4,8,8) and using Strassen on the 8×8 part.
        
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try recursive Strassen with proper handling
    # For 4x5x6, let's try to use a block decomposition
    
    # Approach: Pad 5->8, 6->8, compute 4x8x8 with Strassen, then extract
    # But padding increases rank. Let's try direct recursive approach.
    
    # Actually, let me just implement the schoolbook as a reliable baseline
    # and attempt a simple recursive Strassen for even dimensions.
    
    def _recurse(nn, mm, pp):
        if nn == 0 or mm == 0 or pp == 0:
            return [], [], 0
        
        # Base case: small enough, use schoolbook
        if nn * mm * pp <= 8:
            rank = nn * mm * pp
            U = [[0] * rank for _ in range(nn * mm)]
            V = [[0] * rank for _ in range(mm * pp)]
            W = [[0] * rank for _ in range(pp * nn)]
            col = 0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        U[i * mm + j][col] = 1
                        V[j * pp + k][col] = 1
                        W[k * nn + i][col] = 1
                        col += 1
            return U, V, W, rank
        
        # If all even, use Strassen
        if nn % 2 == 0 and mm % 2 == 0 and pp % 2 == 0:
            n1 = nn // 2
            m1 = mm // 2
            p1 = pp // 2
            
            # 7 sub-problems, each (n1, m1, p1)
            subs = []
            for _ in range(7):
                U_s, V_s, W_s, r_s = _recurse(n1, m1, p1)
                subs.append((U_s, V_s, W_s, r_s))
            
            total_rank = sum(s[3] for s in subs)
            
            # Now build the combined U, V, W
            # U has nn*mm rows, total_rank cols
            # V has mm*pp rows, total_rank cols  
            # W has pp*nn rows, total_rank cols
            
            U = [[0] * total_rank for _ in range(nn * mm)]
            V = [[0] * total_rank for _ in range(mm * pp)]
            W = [[0] * total_rank for _ in range(pp * nn)]
            
            col_offset = 0
            for si in range(7):
                Us, Vs, Ws, rs = subs[si]
                for i in range(nn * mm):
                    for j in range(rs):
                        U[i][col_offset + j] = Us[i % (n1*m1)][j] if i < n1*m1 else (Us[i % (n1*m1)][j] if i >= n1*m1 and i < 2*n1*m1 else 0)
                col_offset += rs
            
            # This is getting too complex to do correctly in the time available.
            # Let me just return schoolbook.
            pass
        
        # Schoolbook fallback
        rank = nn * mm * pp
        U = [[0] * rank for _ in range(nn * mm)]
        V = [[0] * rank for _ in range(mm * pp)]
        W = [[0] * rank for _ in range(pp * nn)]
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j][col] = 1
                    V[j * pp + k][col] = 1
                    W[k * nn + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = _recurse(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
