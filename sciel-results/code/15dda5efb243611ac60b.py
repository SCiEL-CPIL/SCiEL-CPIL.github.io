"""Rewritten program for matrix multiplication 4x4 by 4x7 using Strassen."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    # Strassen's algorithm for 2x2 matrix multiplication
    # Returns (U, V, W) for 2x2 by 2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: A is 2x2, B is 2x2, C = A*B is 2x2
    # A indexed by (i,j) where i=row, j=col
    # B indexed by (j,k) where j=row, k=col
    # C indexed by (i,k) where i=row, k=col
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]

    # The 7 Strassen products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)(B11)
    # P3 = (A11)(B12-B22)
    # P4 = (A22)(B21-B11)
    # P5 = (A11+A12)(B22)
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)

    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # P1: (A11+A22)(B11+B22)
    # A11: U[0,0]=1, A22: U[1,1]=1
    # B11: V[0,0]=1, B22: V[1,1]=1
    # C11: W[0,0], C22: W[1,1]
    U2[0][0] = 1  # A11
    U2[1][0] = 1  # A22
    V2[0][0] = 1  # B11
    V2[1][0] = 1  # B22
    W2[0][0] = 1  # C11
    W2[1][0] = 1  # C22

    # P2: (A21+A22)(B11)
    # A21: U[1,0]=1, A22: U[1,1]=1
    # B11: V[0,0]=1
    # C21: W[0,1] (since C21 is row 2, col 1 -> W[(1)*2+1] = W[3])
    # Wait, W index is k*2+i where k=col of C, i=row of C
    # C21 means i=1 (second row), k=0 (first col) -> W[0*2+1] = W[1]
    # Let me re-index carefully.
    # A is 2x2: A[i,j] for i in {0,1}, j in {0,1}
    # B is 2x2: B[j,k] for j in {0,1}, k in {0,1}
    # C is 2x2: C[i,k] for i in {0,1}, k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # C11: i=0, k=0 -> W[0*2+0] = W[0]
    # C12: i=0, k=1 -> W[1*2+0] = W[2]
    # C21: i=1, k=0 -> W[0*2+1] = W[1]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]

    # Let me redo with correct W indexing
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # P1: (A11+A22)(B11+B22)
    U2[0][0] += 1   # A[0,0]
    U2[3][0] += 1   # A[1,1] (row 1, col 1 -> index 1*2+1=3)
    V2[0][0] += 1   # B[0,0]
    V2[3][0] += 1   # B[1,1] (row 1, col 1 -> index 1*2+1=3)
    W2[0][0] += 1   # C[0,0]
    W2[3][0] += 1   # C[1,1]

    # P2: (A21+A22)(B11)
    U2[2][1] += 1   # A[1,0] (row 1, col 0 -> index 1*2+0=2)
    U2[3][1] += 1   # A[1,1]
    V2[0][1] += 1   # B[0,0]
    W2[1][1] += 1   # C[1,0] (k=0, i=1 -> W[0*2+1]=W[1])

    # P3: (A11)(B12-B22)
    U2[0][2] += 1   # A[0,0]
    V2[1][2] += 1   # B[0,1] (row 0, col 1 -> index 0*2+1=1)
    V2[3][2] -= 1   # B[1,1]
    W2[2][2] += 1   # C[0,1] (k=1, i=0 -> W[1*2+0]=W[2])

    # P4: (A22)(B21-B11)
    U2[3][3] += 1   # A[1,1]
    V2[2][3] += 1   # B[1,0] (row 1, col 0 -> index 1*2+0=2)
    V2[0][3] -= 1   # B[0,0]
    W2[0][3] += 1   # C[0,0]
    W2[1][3] += 1   # C[1,0]

    # P5: (A11+A12)(B22)
    U2[0][4] += 1   # A[0,0]
    U2[1][4] += 1   # A[0,1] (row 0, col 1 -> index 0*2+1=1)
    V2[3][4] += 1   # B[1,1]
    W2[0][4] -= 1   # C[0,0]
    W2[2][4] += 1   # C[0,1]

    # P6: (A21-A11)(B11+B12)
    U2[2][5] += 1   # A[1,0]
    U2[0][5] -= 1   # A[0,0]
    V2[0][5] += 1   # B[0,0]
    V2[1][5] += 1   # B[0,1]
    W2[3][5] += 1   # C[1,1]

    # P7: (A12-A22)(B21+B22)
    U2[1][6] += 1   # A[0,1]
    U2[3][6] -= 1   # A[1,1]
    V2[2][6] += 1   # B[1,0]
    V2[3][6] += 1   # B[1,1]
    W2[0][6] += 1   # C[0,0]
    W2[2][6] += 1   # C[0,1]

    # Now compose two levels of Strassen to get 4x4 by 4x8
    # First level: 2x2 blocks, each 2x2
    # Second level: 2x2 blocks, each 4x4
    # We need 4x4 by 4x8, so we pad B to 4x8

    # Compose Strassen 2x2 twice to get 4x4 by 4x4 (rank 49)
    # Then extend to 4x8 by padding

    # Let's build the 4x4 by 4x8 Strassen decomposition directly.
    # Split A (4x4) into 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]]
    # Split B (4x8) into 2x2 blocks of 2x4:
    # B = [[B00, B01], [B10, B11]]
    # C = A*B is 4x8
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # Apply Strassen to each 2x2 block multiplication where the inner dimensions match.
    # But the blocks are 2x2 by 2x4, not 2x2 by 2x2.
    # This doesn't fit directly.

    # Alternative: Use Strassen for 4x4 by 4x4 (rank 49), then extend to 4x8.
    # For 4x4 by 4x8, we can view it as two 4x4 by 4x4 multiplications (columns 0-3 and 4-7).
    # But that would be rank 49+49=98, which is what we want.

    # Let's build the 4x4 by 4x4 Strassen decomposition first.
    # Compose the 2x2 Strassen decomposition twice.

    # The 4x4 by 4x4 Strassen decomposition has rank 49.
    # We can build it by composing the 2x2 Strassen decomposition.

    # Let's use a more direct approach: build the 4x4 by 4x8 decomposition.
    # Since 4x4 by 4x8 = (4x4 by 4x4) + (4x4 by 4x4) (for the two column halves),
    # the rank is 49 + 49 = 98.

    # But we can do better by using Strassen's algorithm on the 4x4 by 4x8 directly.
    # Actually, Strassen's algorithm works for any size that is a power of 2.
    # 4x4 by 4x8: split into 2x2 blocks where each block is 2x2 by 2x4.
    # This doesn't fit Strassen's 2x2 by 2x2 form.

    # Let's just use the 4x4 by 4x4 Strassen (rank 49) for the first 4 columns,
    # and another 4x4 by 4x4 Strassen (rank 49) for the last 3 columns (with padding to 4).
    # Total rank: 98.

    # Build 4x4 by 4x4 Strassen decomposition.
    # Compose two levels of 2x2 Strassen.

    def compose_strassen(U1, V1, W1, level):
        """Compose Strassen decompositions."""
        if level == 0:
            return U1, V1, W1
        
        # Each U1, V1, W1 is for 2x2 by 2x2
        # We want to build 4x4 by 4x4
        # U: (16, 49), V: (16, 49), W: (16, 49)
        n2 = 4
        rank = 7 * 7
        U = [[0]*rank for _ in range(n2*n2)]
        V = [[0]*rank for _ in range(n2*n2)]
        W = [[0]*rank for _ in range(n2*n2)]
        
        # 7 blocks at the top level
        # Each block is a 2x2 by 2x2 Strassen decomposition
        # The 7 top-level products are:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)(B00)
        # P3 = (A00)(B01-B11)
        # P4 = (A11)(B10-B00)
        # P5 = (A00+A01)(B11)
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Each Pi is a 2x2 by 2x2 product with rank 7
        # So we need to map the indices properly
        
        # A is 4x4, split into 2x2 blocks of 2x2:
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        
        # B is 4x4, split into 2x2 blocks of 2x2:
        # B00: rows 0-1, cols 0-1
        # B01: rows 0-1, cols 2-3
        # B10: rows 2-3, cols 0-1
        # B11: rows 2-3, cols 2-3
        
        # C is 4x4, split into 2x2 blocks of 2x2:
        # C00: rows 0-1, cols 0-1
        # C01: rows 0-1, cols 2-3
        # C10: rows 2-3, cols 0-1
        # C11: rows 2-3, cols 2-3
        
        # For each top-level product Pi, we use the 2x2 Strassen decomposition
        # The U, V, W matrices for the 2x2 Strassen are U2, V2, W2
        
        # Top-level P1: (A00+A11)(B00+B11)
        # A00+A11: U entries for A00 and A11
        # B00+B11: V entries for B00 and B11
        # C00, C11: W entries for C00 and C11
        
        # The 2x2 Strassen decomposition U2 has 4 rows (for 2x2 matrix)
        # Row 0: A[0,0], Row 1: A[0,1], Row 2: A[1,0], Row 3: A[1,1]
        # Similarly for V2 and W2
        
        # For P1, we need to combine A00 and A11 (each 2x2)
        # A00 maps to rows 0-3 of the 4x4 A
        # A11 maps to rows 4-7 of the 4x4 A
        
        # This is getting complex. Let me use a simpler approach.
        
        # I'll directly construct the 4x4 by 4x8 decomposition using the known
        # Strassen-based rank-98 decomposition.
        
        pass
    
    # Let me just directly build the rank-98 decomposition.
    # Use Strassen for 4x4 by 4x4 (rank 49) twice: once for cols 0-3, once for cols 4-7.
    
    # Build 4x4 by 4x4 Strassen decomposition by composing 2x2 Strassen twice.
    
    # Level 1: 2x2 Strassen (rank 7)
    # Level 2: Compose 2x2 Strassen to get 4x4 Strassen (rank 49)
    
    # The 4x4 by 4x4 Strassen decomposition:
    # U: (16, 49), V: (16, 49), W: (16, 49)
    
    # Let's build it step by step.
    
    # First, define the 2x2 Strassen decomposition as above (U2, V2, W2 with rank 7)
    
    # Now compose to get 4x4 by 4x4.
    # The 4x4 by 4x4 Strassen has 7 top-level products, each of rank 7.
    
    # U4: (16, 49)
    # For each top-level product t (0-6), and each sub-product s (0-6):
    # Column t*7+s
    
    # Top-level products:
    # T0: (A00+A11)(B00+B11) -> contributes to C00, C11
    # T1: (A10+A11)(B00) -> contributes to C10
    # T2: (A00)(B01-B11) -> contributes to C01
    # T3: (A11)(B10-B00) -> contributes to C00, C10
    # T4: (A00+A01)(B11) -> contributes to C00, C01
    # T5: (A10-A00)(B00+B01) -> contributes to C11
    # T6: (A01-A11)(B10+B11) -> contributes to C00, C01
    
    # Wait, I need to be more careful. Let me use the standard Strassen formulas.
    
    # For 4x4 by 4x4:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen's algorithm for the 2x2 block multiplication:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B01-B11)
    # P4 = (A11)(B10-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 by 2x2 product, which has a rank-7 decomposition.
    
    # So the total rank is 7*7 = 49.
    
    # Now I need to build U4, V4, W4 for the 4x4 by 4x4 case.
    
    # A is 4x4. Block A00: rows 0-1, cols 0-1. Block A01: rows 0-1, cols 2-3.
    # Block A10: rows 2-3, cols 0-1. Block A11: rows 2-3, cols 2-3.
    
    # U4[i*4+j, c] where i,j in 0-3
    
    # For P1: (A00+A11)(B00+B11)
    # A00+A11: U entries for A00 (rows 0-3) and A11 (rows 4-7)
    # B00+B11: V entries for B00 (rows 0-3) and B11 (rows 4-7)
    # C00: W entries for rows 0-3, C11: W entries for rows 4-7
    
    # The 2x2 Strassen decomposition U2 has 4 rows.
    # U2[0] corresponds to A[0,0], U2[1] to A[0,1], U2[2] to A[1,0], U2[3] to A[1,1]
    
    # For P1, the "A" matrix is A00+A11 (4x4), and the "B" matrix is B00+B11 (4x4).
    # Wait, no. P1 is (A00+A11)(B00+B11) where A00, A11, B00, B11 are 2x2.
    # So P1 is a 2x2 by 2x2 product.
    
    # The result of P1 is a 2x2 matrix, which contributes to C00 and C11.
    
    # So for P1, we use the 2x2 Strassen decomposition U2, V2, W2.
    # The U2 entries map to the 2x2 matrix A00+A11.
    # A00 is at rows 0-1, cols 0-1 of A4x4.
    # A11 is at rows 2-3, cols 2-3 of A4x4.
    
    # So U2[0] (A[0,0]) maps to U4[0, c] (row 0, col 0 of A4x4)
    # U2[1] (A[0,1]) maps to U4[1, c] (row 0, col 1 of A4x4)
    # U2[2] (A[1,0]) maps to U4[4, c] (row 1, col 0 of A4x4)
    # U2[3] (A[1,1]) maps to U4[5, c] (row 1, col 1 of A4x4)
    
    # Wait, A4x4 is indexed as A[i,j] for i in 0-3, j in 0-3.
    # U4[i*4+j, c]
    # A00: i in 0-1, j in 0-1 -> U4 indices 0,1,4,5
    # A01: i in 0-1, j in 2-3 -> U4 indices 2,3,6,7
    # A10: i in 2-3, j in 0-1 -> U4 indices 8,9,12,13
    # A11: i in 2-3, j in 2-3 -> U4 indices 10,11,14,15
    
    # For P1: (A00+A11)(B00+B11)
    # A00+A11: U4 entries for A00 and A11
    # U2[0] (A[0,0] of the 2x2) -> A00[0,0] -> U4[0, c]
    # U2[1] (A[0,1] of the 2x2) -> A00[0,1] -> U4[1, c]
    # U2[2] (A[1,0] of the 2x2) -> A00[1,0] -> U4[4, c]
    # U2[3] (A[1,1] of the 2x2) -> A00[1,1] -> U4[5, c]
    # Also A11[0,0] -> U4[10, c], A11[0,1] -> U4[11, c], A11[1,0] -> U4[14, c], A11[1,1] -> U4[15, c]
    
    # So for P1, U4[0, c] += U2[0, s], U4[1, c] += U2[1, s], etc.
    # And similarly for A11.
    
    # This is getting very involved. Let me just use a simpler approach.
    
    # I'll directly construct the rank-98 decomposition for 4x4 by 4x8.
    # Use two copies of the 4x4 by 4x4 Strassen decomposition (rank 49 each).
    # First copy: for B columns 0-3.
    # Second copy: for B columns 4-7 (but we only need 4-6, so pad column 7 with 0).
    
    # Build 4x4 by 4x4 Strassen decomposition.
    
    # I'll use a recursive approach to build the Strassen decomposition.
    
    def build_strassen(n):
        """Build Strassen decomposition for n x n by n x n.
        n must be a power of 2.
        Returns (U, V, W) where U is (n*n, rank), V is (n*n, rank), W is (n*n, rank).
        """
        if n == 2:
            # Return the 2x2 Strassen decomposition
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # P1: (A11+A22)(B11+B22)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # P2: (A21+A22)(B11)
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[1][1] = 1
            
            # P3: (A11)(B12-B22)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[2][2] = 1
            
            # P4: (A22)(B21-B11)
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[1][3] = 1
            
            # P5: (A11+A12)(B22)
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[2][4] = 1
            
            # P6: (A21-A11)(B11+B12)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # P7: (A12-A22)(B21+B22)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[2][6] = 1
            
            return U, V, W
        
        # Recursive case
        n_half = n // 2
        U1, V1, W1 = build_strassen(n_half)
        
        # U1, V1, W1 are for n_half x n_half by n_half x n_half
        # We want to build n x n by n x n
        
        rank = 7 * 7  # For n=4, rank=49
        # Actually, the rank scales as 7^(log2(n)) = n^(log2(7))
        # For n=4, rank = 7^2 = 49
        
        U = [[0]*rank for _ in range(n*n)]
        V = [[0]*rank for _ in range(n*n)]
        W = [[0]*rank for _ in range(n*n)]
        
        # The 7 top-level products
        # T0: (A00+A11)(B00+B11)
        # T1: (A10+A11)(B00)
        # T2: (A00)(B01-B11)
        # T3: (A11)(B10-B00)
        # T4: (A00+A01)(B11)
        # T5: (A10-A00)(B00+B01)
        # T6: (A01-A11)(B10+B11)
        
        # Each Ti is a n_half x n_half by n_half x n_half product
        # So we use U1, V1, W1 for each
        
        # A is n x n, split into 2x2 blocks of n_half x n_half
        # A00: rows 0 to n_half-1, cols 0 to n_half-1
        # A01: rows 0 to n_half-1, cols n_half to 2*n_half-1
        # A10: rows n_half to 2*n_half-1, cols 0 to n_half-1
        # A11: rows n_half to 2*n_half-1, cols n_half to 2*n_half-1
        
        # U[i*n+j, c] for A[i,j]
        
        # For T0: (A00+A11)(B00+B11)
        # A00+A11: U entries for A00 and A11
        # B00+B11: V entries for B00 and B11
        # C00, C11: W entries
        
        # Map U1 rows to U rows
        # U1[i*n_half+j, s] -> U[i*n+j, c] for A00
        # U1[i*n_half+j, s] -> U[(i+n_half)*n + (j+n_half), c] for A11
        
        for s in range(7):  # Sub-product index
            c = 0 * 7 + s  # Top-level product 0
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[i*n + j][c] += val
                    U[(i+n_half)*n + (j+n_half)][c] += val
            
            c = 0 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[i*n + j][c] += val
                    V[(i+n_half)*n + (j+n_half)][c] += val
            
            # W: C00 and C11
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[i*n + j][c] += val
                    W[(i+n_half)*n + (j+n_half)][c] += val
        
        # T1: (A10+A11)(B00)
        # A10+A11: U entries for A10 and A11
        # B00: V entries for B00
        # C10: W entries
        
        for s in range(7):
            c = 1 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[(i+n_half)*n + j][c] += val  # A10
                    U[(i+n_half)*n + (j+n_half)][c] += val  # A11
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[i*n + j][c] += val  # B00
            
            # W: C10
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[(i+n_half)*n + j][c] += val
        
        # T2: (A00)(B01-B11)
        # A00: U entries for A00
        # B01-B11: V entries for B01 and -B11
        # C01: W entries
        
        for s in range(7):
            c = 2 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[i*n + j][c] += val  # A00
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[i*n + (j+n_half)][c] += val  # B01
                    V[(i+n_half)*n + (j+n_half)][c] -= val  # -B11
            
            # W: C01
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[i*n + (j+n_half)][c] += val
        
        # T3: (A11)(B10-B00)
        # A11: U entries for A11
        # B10-B00: V entries for B10 and -B00
        # C00, C10: W entries
        
        for s in range(7):
            c = 3 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[(i+n_half)*n + (j+n_half)][c] += val  # A11
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[(i+n_half)*n + j][c] += val  # B10
                    V[i*n + j][c] -= val  # -B00
            
            # W: C00 and C10
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[i*n + j][c] += val  # C00
                    W[(i+n_half)*n + j][c] += val  # C10
        
        # T4: (A00+A01)(B11)
        # A00+A01: U entries for A00 and A01
        # B11: V entries for B11
        # C00, C01: W entries
        
        for s in range(7):
            c = 4 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[i*n + j][c] += val  # A00
                    U[i*n + (j+n_half)][c] += val  # A01
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[(i+n_half)*n + (j+n_half)][c] += val  # B11
            
            # W: C00 and C01
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[i*n + j][c] -= val  # C00 (note: -P5 in C00 formula)
                    W[i*n + (j+n_half)][c] += val  # C01
        
        # T5: (A10-A00)(B00+B01)
        # A10-A00: U entries for A10 and -A00
        # B00+B01: V entries for B00 and B01
        # C11: W entries
        
        for s in range(7):
            c = 5 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[(i+n_half)*n + j][c] += val  # A10
                    U[i*n + j][c] -= val  # -A00
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[i*n + j][c] += val  # B00
                    V[i*n + (j+n_half)][c] += val  # B01
            
            # W: C11
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[(i+n_half)*n + (j+n_half)][c] += val
        
        # T6: (A01-A11)(B10+B11)
        # A01-A11: U entries for A01 and -A11
        # B10+B11: V entries for B10 and B11
        # C00, C01: W entries
        
        for s in range(7):
            c = 6 * 7 + s
            for i in range(n_half):
                for j in range(n_half):
                    val = U1[i*n_half + j][s]
                    U[i*n + (j+n_half)][c] += val  # A01
                    U[(i+n_half)*n + (j+n_half)][c] -= val  # -A11
            
            for i in range(n_half):
                for j in range(n_half):
                    val = V1[i*n_half + j][s]
                    V[(i+n_half)*n + j][c] += val  # B10
                    V[(i+n_half)*n + (j+n_half)][c] += val  # B11
            
            # W: C00 and C01
            for i in range(n_half):
                for j in range(n_half):
                    val = W1[i*n_half + j][s]
                    W[i*n + j][c] += val  # C00
                    W[i*n + (j+n_half)][c] += val  # C01
        
        return U, V, W
    
    # Build 4x4 by 4x4 Strassen decomposition
    U4, V4, W4 = build_strassen(4)
    
    # Now extend to 4x4 by 4x8
    # We need two copies: one for B cols 0-3, one for B cols 4-7
    # But we only need B cols 4-6 (3 columns), so we pad col 7 with 0
    
    # U: (16, 98), V: (28, 98), W: (28, 98)
    # U is the same for both copies
    # V has 28 rows (4*7), W has 28 rows (7*4)
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # First copy: B cols 0-3
    # V4 has 16 rows (4*4), but we only need the first 16 rows for B cols 0-3
    # Wait, V4 is for 4x4 B, so V4[j*4+k, c] for B[j,k]
    # For B cols 0-3, V4[j*4+k, c] maps to V[j*7+k, c] for k in 0-3
    
    for c in range(49):
        for i in range(16):
            U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                V[j*7 + k][c] = V4[j*4 + k][c]
        # W4 has 16 rows (4*4), W4[k*4+i, c] for C[i,k]
        # For B cols 0-3, W4[k*4+i, c] maps to W[k*4+i, c] for k in 0-3
        for k in range(4):
            for i in range(4):
                W[k*4 + i][c] = W4[k*4 + i][c]
    
    # Second copy: B cols 4-7 (but we only need 4-6)
    # V4[j*4+k, c] maps to V[j*7+(k+4), c] for k in 0-3
    # But we only need k in 0-2 (cols 4-6), so k=3 (col 7) is padded with 0
    
    for c in range(49, 98):
        c1 = c - 49
        for i in range(16):
            U[i][c] = U4[i][c1]
        for j in range(4):
            for k in range(3):  # Only cols 4-6
                V[j*7 + (k+4)][c] = V4[j*4 + k][c1]
            # V[j*7 + 7] is not used (col 7 is padded)
        for k in range(4):
            for i in range(4):
                if k < 4:  # All k in 0-3, but we only have 7 cols in output
                    # W[k*4+i, c] for k in 0-3
                    W[k*4 + i][c] = W4[k*4 + i][c1]
    
    # But wait, the output C is 4x7, so W should have 28 rows (7*4).
    # W[k*4+i, c] where k is col of C (0-6), i is row of C (0-3)
    # For the second copy, k in 0-3 maps to cols 4-7 of C
    # But we only have cols 0-6, so k=3 (col 7) should be dropped
    
    # Let me fix this. The W matrix for the second copy should only include
    # cols 4-6, i.e., k in 0-2
    
    # Actually, I need to be more careful. Let me redo this.
    
    # The output C is 4x7. C[i,k] for i in 0-3, k in 0-6.
    # W[k*4+i, c]
    
    # First copy (B cols 0-3): C[i,k] for k in 0-3
    # W4[k*4+i, c] maps to W[k*4+i, c]
    
    # Second copy (B cols 4-7): C[i,k] for k in 4-7
    # But we only have k in 4-6, so k=7 is dropped
    # W4[k*4+i, c] for k in 0-2 maps to W[(k+4)*4+i, c]
    # W4[k*4+i, c] for k=3 is dropped
    
    # Let me redo the construction more carefully.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # First copy: B cols 0-3
    for c in range(49):
        for i in range(16):
            U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                V[j*7 + k][c] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][c] = W4[k*4 + i][c]
    
    # Second copy: B cols 4-6 (3 columns)
    for c in range(49, 98):
        c1 = c - 49
        for i in range(16):
            U[i][c] = U4[i][c1]
        for j in range(4):
            for k in range(3):  # B cols 4-6
                V[j*7 + (k+4)][c] = V4[j*4 + k][c1]
        for k in range(3):  # C cols 4-6
            for i in range(4):
                W[(k+4)*4 + i][c] = W4[k*4 + i][c1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
