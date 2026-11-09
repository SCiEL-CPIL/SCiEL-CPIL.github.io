"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Recursive Strassen decomposition for matrix multiplication
    # We build a rank decomposition for n x m * m x p matrix multiplication
    # using recursive Strassen splitting
    
    def strassen_decomposition(n, m, p):
        """
        Build a rank decomposition for multiplying an n x m matrix by an m x p matrix.
        Returns (U, V, W) where:
        - U has shape (n*m, rank)
        - V has shape (m*p, rank)  
        - W has shape (p*n, rank)
        
        Uses recursive Strassen: split into halves, use 7 sub-products.
        Base case: 1x1 x 1x1 uses rank 1.
        """
        # Base case: 1x1 x 1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # Split dimensions
        n1, n2 = n // 2, n - n // 2
        m1, m2 = m // 2, m - m // 2
        p1, p2 = p // 2, p - p // 2
        
        # We need to handle non-square splits carefully
        # For simplicity, require even dimensions or handle odd specially
        # Let's assume we can always split (handle edge cases)
        
        if n1 == 0 or m1 == 0 or p1 == 0:
            # Degenerate case, use schoolbook
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
        
        # The 7 Strassen products:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # Each Pi is a matrix multiplication of size (n1 x m1) * (m1 x p1) etc.
        # But the dimensions vary. Let me think about this more carefully.
        
        # Actually, for the recursive Strassen to work cleanly, we need:
        # A is n x m, split into:
        #   A11: n1 x m1, A12: n1 x m2
        #   A21: n2 x m1, A22: n2 x m2
        # B is m x p, split into:
        #   B11: m1 x p1, B12: m1 x p2
        #   B21: m2 x p1, B22: m2 x p2
        # C is n x p, split into:
        #   C11: n1 x p1, C12: n1 x p2
        #   C21: n2 x p1, C22: n2 x p2
        
        # The 7 products have different input dimensions:
        # P1: (n1 x m1 + n2 x m2) * (m1 x p1 + m2 x p2) -> not a standard matmul
        
        # This approach is getting complicated with non-square blocks.
        # Let me use a simpler approach: just use schoolbook for the full 4x4 x 4x5.
        
        # Actually, let me try a different strategy: use the known fact that
        # 4x4 x 4x4 has rank 49 (via Strassen), and add the 4x4 x 4x1 part.
        
        # For now, let me just return schoolbook to ensure correctness.
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
    
    # For 4x4 x 4x5, let me try to use a 49-rank decomposition for the 4x4 x 4x4 part
    # plus 16 terms for the 4x4 x 4x1 part, total 65.
    
    # But first, let me implement the 49-rank Strassen for 4x4 x 4x4 properly.
    
    # I'll use a helper that builds the Strassen decomposition for 2x2 x 2x2
    # and then compose it.
    
    def strassen_2x2():
        """
        Return (U, V, W) for 2x2 x 2x2 matrix multiplication using Strassen.
        U: 4 x 7, V: 4 x 7, W: 4 x 7
        """
        # Strassen's 7 products for 2x2 x 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Result:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = -1; W[1][0] = 1; W[2][0] = -1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1
        
        return U, V, W
    
    # Now for 4x4 x 4x4, we split into 2x2 blocks and use Strassen at the block level.
    # Then each block product is a 2x2 x 2x2 multiplication, which we decompose with Strassen.
    
    # Let me build this systematically.
    
    # A is 4x4, split into:
    # A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
    # A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    
    # B is 4x4, split into:
    # B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
    # B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    
    # C is 4x4, split into:
    # C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3)
    # C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # The 7 block-level Strassen products:
    # P1 = (A11+A22)(B11+B22) -> 2x2 x 2x2
    # P2 = (A21+A22)B11 -> 2x2 x 2x2
    # P3 = A11(B12-B22) -> 2x2 x 2x2
    # P4 = A22(B21-B11) -> 2x2 x 2x2
    # P5 = (A11+A12)B22 -> 2x2 x 2x2
    # P6 = (A21-A11)(B11+B12) -> 2x2 x 2x2
    # P7 = (A12-A22)(B21+B22) -> 2x2 x 2x2
    
    # Each Pi is computed with 7 scalar multiplications (Strassen for 2x2 x 2x2)
    
    # Total: 7 * 7 = 49 scalar multiplications
    
    # Now I need to express each Pi's inputs as linear combinations of the original A and B entries.
    
    # Let me define helper functions to extract block entries.
    
    # A[i,j] where i,j in 0..3
    # Block A11: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5 in row-major (i*4+j)
    # Block A12: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
    # Block A21: A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8,9,12,13
    # Block A22: A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10,11,14,15
    
    # Similarly for B (4x4):
    # B[j,k] where j,k in 0..3
    # Block B11: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,4,5
    # Block B12: B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,6,7
    # Block B21: B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8,9,12,13
    # Block B22: B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10,11,14,15
    
    # For C (4x4), using W[k*n+i] convention where n=4:
    # C[i,k] -> W[k*4+i]
    # Block C11: C[0,0], C[0,1], C[1,0], C[1,1] -> W[0], W[1], W[4], W[5]
    # Block C12: C[0,2], C[0,3], C[1,2], C[1,3] -> W[2], W[3], W[6], W[7]
    # Block C21: C[2,0], C[2,1], C[3,0], C[3,1] -> W[8], W[9], W[12], W[13]
    # Block C22: C[2,2], C[2,3], C[3,2], C[3,3] -> W[10], W[11], W[14], W[15]
    
    # This is getting very complex. Let me just use schoolbook for now to ensure correctness.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
