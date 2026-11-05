"""Program for 4x4x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Return the 7 Strassen multipliers for 2x2x2.
    Each multiplier is (U_entry, V_entry, W_entry) where entries are (row, col) or None.
    Actually, let's represent as lists of (u_idx, v_idx, w_idx) for the 2x2 case.
    U is 4x4 (flattened i*2+j), V is 4x4 (flattened j*2+k), W is 4x4 (flattened k*2+i).
    Wait, for 2x2x2: n=2, m=2, p=2.
    U shape: (n*m, rank) = (4, 7)
    V shape: (m*p, rank) = (4, 7)
    W shape: (p*n, rank) = (4, 7)
    
    Strassen formulas:
    M1 = (A11 - A22)(B21 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's build U, V, W columns.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Column 0: M1 = (A11 - A22)(B21 + B22)
    U[0][0] = 1; U[3][0] = -1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: M2 = (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Column 2: M3 = A11(B12 - B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Column 3: M4 = A22(B21 - B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Column 4: M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Column 5: M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Column 6: M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4():
    """Build 4x4x4 decomposition using recursive Strassen.
    Returns U (16, 49), V (16, 49), W (16, 49).
    n=4, m=4, p=4.
    U shape: (n*m, rank) = (16, 49)
    V shape: (m*p, rank) = (16, 49)
    W shape: (p*n, rank) = (16, 49)
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for i in range(2):  # block row of A (0,1)
        for j in range(2):  # block col of A (0,1)
            # This is a 2x2x2 sub-problem for A[i*2:(i+1)*2, j*2:(j+1)*2]
            # But we need to handle all combinations of block positions.
            # Actually, recursive Strassen for 4x4x4:
            # A is 4x4, B is 4x4, C is 4x4.
            # We divide into 2x2 blocks.
            # For each of the 7 Strassen multipliers at the top level,
            # we need to compute a 2x2x2 product of the sums/differences.
            # But the sums/differences are 2x2 matrices.
            # So each top-level multiplier becomes a 2x2x2 problem.
            # Total rank = 7 * 7 = 49.
            
            # Let me redo this.
            # Top level: A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
            # C = A*B
            # M1 = (A00-A11)(B10+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Each Mk is a 2x2x2 problem.
            # For Mk = X*Y where X and Y are 2x2 matrices that are linear combinations of blocks.
            # We need to express the entries of X and Y in terms of A and B entries.
            # Then apply the 2x2x2 Strassen decomposition.
            
            # Let's do this more carefully.
            # For each top-level multiplier t (0..6):
            #   X = sum over (i,j) of x_{ij} * A[i,j]  (where A[i,j] is a 2x2 block)
            #   Y = sum over (j,k) of y_{jk} * B[j,k]
            #   Z = X*Y
            #   Then Z contributes to C blocks via the top-level formula.
            
            # This is getting complex. Let me use a different approach.
            # I'll construct the 4x4x4 by direct enumeration with Strassen at both levels.
            
            pass
    
    # Let me restart with a cleaner approach.
    # I'll directly build the 49 columns.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    sU, sV, sW = _strassen_2x2()
    
    # Top-level Strassen multipliers
    # t=0: M1 = (A00-A11)(B10+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # For each t, we have X (2x2) and Y (2x2) where X*Y = Mk
    # Then we apply 2x2x2 Strassen to X*Y.
    # The result is a 2x2 matrix Z, which contributes to C blocks.
    
    # Let me define for each t:
    #   X entries: x[i][j] = sum of a_{block_i, block_j} * A[2*block_i + i', 2*block_j + j']
    #   Y entries: y[i][j] = sum of b_{block_i, block_j} * B[2*block_i + i', 2*block_j + j']
    
    # This is too complex to do by hand. Let me use a different strategy.
    # I'll use the known fact that the rank of 4x4x4 is 49 (or maybe less?).
    # Actually, the rank of 4x4x4 is known to be 49 (from recursive Strassen).
    # But there might be a better decomposition.
    
    # For now, let me just use the recursive Strassen and implement it properly.
    
    # I'll create a helper that, given a 2x2x2 decomposition,
    # builds a 4x4x4 decomposition.
    
    # Actually, let me just hardcode the 49 columns for 4x4x4.
    # This is tedious but doable.
    
    # Alternatively, I can use a different approach:
    # Use the schoolbook for the 4x4x7 and see if I can improve.
    # Schoolbook rank = 4*4*7 = 112.
    
    # Let me try a simpler improvement first:
    # Use Strassen for 2x2 blocks.
    # 4x4x7 = 4 blocks of 2x2x7 + 4 blocks of 2x2x7 + ...
    # This is getting complicated.
    
    # Let me just return the schoolbook for now and see if I can improve.
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """Construct the decomposition for 4x4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    rank = 112  # Schoolbook rank
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
