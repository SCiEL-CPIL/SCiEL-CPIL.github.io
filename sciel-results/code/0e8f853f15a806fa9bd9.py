"""Rewritten program: block-based tensor decomposition for 3x5x5 matrix multiplication.
Uses recursive Strassen-style splitting with memoization to find a low-rank decomposition,
then explicitly constructs the factor matrices.
"""

DIMENSIONS = [3, 5, 5]

def _strassen_factors(n, m, p):
    """Return (U, V, W) factor matrices for n x m x p matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    Using schoolbook or Strassen decomposition.
    """
    import numpy as np
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return (np.array([[1]]), np.array([[1]]), np.array([[1]]))
    
    if n == 2 and m == 2 and p == 2:
        # Strassen 2x2x2
        # U factors for A (2x2), V factors for B (2x2), W factors for C (2x2)
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B12+B22)
        # M7 = (A12-A22)(B11+B21)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        #
        # U (4x7): rows indexed by (i,j) for A[i,j], col = k for Mk
        # V (4x7): rows indexed by (j,k) for B[j,k], col = k for Mk
        # W (4x7): rows indexed by (k,i) for C[k,i], col = k for Mk
        #
        # A indices: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
        # B indices: (0,0)=B00, (0,1)=B01, (1,0)=B10, (1,1)=B11
        # C indices: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
        #
        # M1: A00,A11 with B00,B11
        # M2: A10,A11 with B00
        # M3: A00 with B10,B11
        # M4: A11 with B00,B01
        # M5: A00,A01 with B11
        # M6: A10,A00 with B01,B11
        # M7: A01,A11 with B00,B10
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Row indices: A: 00=0, 01=1, 10=2, 11=3
        #             B: 00=0, 01=1, 10=2, 11=3
        #             C: 00=0, 01=1, 10=2, 11=3
        # M1: col 0
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        # M2: col 1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        # M3: col 2
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[1,2] = 1
        # M4: col 3
        U[3,3] = 1
        V[0,3] = 1; V[1,3] = -1
        W[0,3] = 1; W[2,3] = 1
        # M5: col 4
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        # M6: col 5
        U[2,5] = 1; U[0,5] = -1
        V[1,5] = 1; V[3,5] = 1
        W[3,5] = 1
        # M7: col 6
        U[1,6] = 1; U[3,6] = -1
        V[0,6] = 1; V[2,6] = 1
        W[0,6] = 1
        return (U, V, W)
    
    # General case: use schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return (U, V, W)


def _combine_factors(factors_list, splits):
    """Combine factor matrices from sub-problems.
    factors_list: list of (U, V, W) tuples
    splits: list of (n_i, m_i, p_i) for each sub-problem
    Returns combined (U, V, W)
    """
    import numpy as np
    total_rank = sum(U.shape[1] for U, V, W in factors_list)
    # We need to figure out the dimensions from the splits
    # Actually, this is getting complicated. Let me use a different approach.
    pass


def construct():
    """Construct a low-rank decomposition for 3x5x5 matrix multiplication.
    
    Strategy: Split the 5-dimension into 2+3, and use known optimal decompositions
    for the resulting sub-problems.
    
    3x5x5 = (3x2x5) + (3x3x5) where we split the middle dimension.
    Actually, C_{3,5} = A_{3,5} * B_{5,5}.
    Split A's columns: A = [A_left (3x2), A_right (3x3)]
    Split B's rows: B = [B_top (2x5); B_bottom (3x5)]
    C = A_left * B_top + A_right * B_bottom
    
    So we need M_{3,2,5} and M_{3,3,5}.
    
    For M_{3,2,5}: split 5 into 2+2+1
    C_{3,2} = A_{3,2} * B_{2,2} + A_{3,2} * B_{2,2} + A_{3,2} * B_{2,1}
    = M_{3,2,2} + M_{3,2,2} + M_{3,2,1}
    
    For M_{3,3,5}: split 5 into 2+3
    C_{3,3} = A_{3,3} * B_{3,2} + A_{3,3} * B_{3,3}
    = M_{3,3,2} + M_{3,3,3}
    
    We need explicit factorizations for:
    M_{3,2,2}, M_{3,2,1}, M_{3,3,2}, M_{3,3,3}
    
    Let me implement each using schoolbook or Strassen where applicable.
    """
    import numpy as np
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[1,2] = 1
        U[3,3] = 1
        V[0,3] = 1; V[1,3] = -1
        W[0,3] = 1; W[2,3] = 1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        U[2,5] = 1; U[0,5] = -1
        V[1,5] = 1; V[3,5] = 1
        W[3,5] = 1
        U[1,6] = 1; U[3,6] = -1
        V[0,6] = 1; V[2,6] = 1
        W[0,6] = 1
        return U, V, W
    
    def M_322():
        """3x2x2: split 3=1+2, use 1x2x2 (schoolbook rank 4) + 2x2x2 (Strassen rank 7) = 11"""
        U1, V1, W1 = schoolbook(1, 2, 2)  # rank 4
        U2, V2, W2 = strassen_222()       # rank 7
        # Combine: need to interleave rows
        # A is 3x2: rows 0,1,2. Split: row 0 (1x2), rows 1-2 (2x2)
        # B is 2x2: same for both
        # C is 2x3: cols 0,1,2. Split: col 0 (2x1), cols 1-2 (2x2)
        #
        # U: 6 rows (3*2), total rank 11
        U = np.zeros((6, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))
        # Part 1: A[0,:], B, C[:,0] -> rank 4
        U[0:2, 0:4] = U1  # rows 0,1 of A (which is row 0 of A, cols 0,1)
        V[0:4, 0:4] = V1
        W[0:2, 0:4] = W1  # C col 0, rows 0,1
        # Part 2: A[1:3,:], B, C[:,1:3] -> rank 7
        U[2:6, 4:11] = U2
        V[0:4, 4:11] = V2
        W[2:6, 4:11] = W2  # C cols 1,2, rows 0,1
        return U, V, W
    
    def M_321():
        """3x2x1: schoolbook, rank 6"""
        return schoolbook(3, 2, 1)
    
    def M_332():
        """3x3x2: split 3=1+2 for first dim, 2=1+1 for third dim.
        This is complex. Let's use schoolbook for now: rank 18.
        Actually, let's try: split first dim 3=1+2, third dim 2=1+1.
        4 sub-problems: M_{1,3,1}, M_{1,3,1}, M_{2,3,1}, M_{2,3,1}
        M_{1,3,1} rank 3, M_{2,3,1} rank 6. Total = 3+3+6+6 = 18. Same as schoolbook.
        Try splitting second dim: 3=1+2.
        M_{3,1,2} + M_{3,2,2} = 6 + 11 = 17. Better!
        """
        U1, V1, W1 = schoolbook(3, 1, 2)   # rank 6
        U2, V2, W2 = M_322()               # rank 11
        # A is 3x3: split cols: col 0 (3x1), cols 1-2 (3x2)
        # B is 3x2: split rows: row 0 (1x2), rows 1-2 (2x2)
        # C is 2x3: same
        U = np.zeros((9, 17))
        V = np.zeros((6, 17))
        W = np.zeros((6, 17))
        U[0:3, 0:6] = U1
        V[0:2, 0:6] = V1
        W[0:6, 0:6] = W1
        U[3:9, 6:17] = U2
        V[2:6, 6:17] = V2
        W[0:6, 6:17] = W2
        return U, V, W
    
    def M_333():
        """3x3x3: Use known rank 23 decomposition.
        For simplicity, use schoolbook rank 27 for now, or implement Bürgisser et al.
        Let's use schoolbook: rank 27.
        """
        return schoolbook(3, 3, 3)
    
    # Now combine for 3x5x5
    # Split middle dim 5 = 2 + 3
    # C_{3,5} = A_{3,2} * B_{2,5} + A_{3,3} * B_{3,5}
    # B_{2,5}: split cols 5 = 2+2+1
    #   B_{2,2}, B_{2,2}, B_{2,1}
    # B_{3,5}: split cols 5 = 2+3
    #   B_{3,2}, B_{3,3}
    
    # Sub-problems:
    # 1. M_{3,2,2} (A_{3,2} * B_{2,2}) -> rank 11
    # 2. M_{3,2,2} (A_{3,2} * B_{2,2}) -> rank 11
    # 3. M_{3,2,1} (A_{3,2} * B_{2,1}) -> rank 6
    # 4. M_{3,3,2} (A_{3,3} * B_{3,2}) -> rank 17
    # 5. M_{3,3,3} (A_{3,3} * B_{3,3}) -> rank 27 (schoolbook) or 23 (optimal)
    
    U1, V1, W1 = M_322()    # rank 11
    U2, V2, W2 = M_322()    # rank 11
    U3, V3, W3 = M_321()    # rank 6
    U4, V4, W4 = M_332()    # rank 17
    U5, V5, W5 = M_333()    # rank 27
    
    total_rank = 11 + 11 + 6 + 17 + 27  # = 72
    # This is only slightly better than schoolbook 75. Let me try to improve.
    
    # Better: use M_{3,3,3} rank 23 (hardcoded or computed)
    # For now, let's just use this and see.
    
    U = np.zeros((15, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    # A is 3x5: rows (i,j) where i in 0..2, j in 0..4
    # B is 5x5: rows (j,k) where j in 0..4, k in 0..4
    # C is 3x5: rows (k,i) where k in 0..4, i in 0..2
    
    # Part 1: A[0:3, 0:2] (3x2), B[0:2, 0:2] (2x2), C[0:2, 0:3] (2x3)
    U[0:6, col:col+11] = U1
    V[0:4, col:col+11] = V1
    W[0:6, col:col+11] = W1
    col += 11
    
    # Part 2: A[0:3, 0:2] (3x2), B[0:2, 2:4] (2x2), C[2:4, 0:3] (2x3)
    U[0:6, col:col+11] = U2
    V[0:4, col:col+11] = V2  # B rows 0,1 -> need to offset in V
    # V rows for B[0:2, 2:4]: row indices (0,2),(0,3),(1,2),(1,3) -> V rows 2,3,6,7
    # Actually V is indexed by j*p+k. For p=5, B[0,2]=V[0*5+2]=V[2], B[0,3]=V[3], B[1,2]=V[7], B[1,3]=V[8]
    # Wait, V shape is (m*p, rank) = (25, rank). V[j*p+k] where j is row of B, k is col of B.
    # For B[0:2, 2:4], j in 0,1; k in 2,3.
    # V rows: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # So we need to place V2 (which is for a 2x2 B) into V rows 2,3,7,8
    V[2, col:col+11] = V2[0, :]
    V[3, col:col+11] = V2[1, :]
    V[7, col:col+11] = V2[2, :]
    V[8, col:col+11] = V2[3, :]
    # W for C[2:4, 0:3]: k in 2,3; i in 0,1,2
    # W[k*n+i] where n=3. k=2,i=0->W[6], k=2,i=1->W[7], k=2,i=2->W[8], k=3,i=0->W[9], etc.
    # W1/W2 are for C with n=3, p=2. W[k*n+i] for k in 0,1 and i in 0,1,2.
    # We need to shift k by 2: k' = k+2.
    W[6:12, col:col+11] = W2
    col += 11
    
    # Part 3: A[0:3, 0:2] (3x2), B[0:2, 4:5] (2x1), C[4:5, 0:3] (1x3)
    U[0:6, col:col+6] = U3
    # V for B[0:2, 4:5]: j in 0,1; k=4. V[0*5+4]=V[4], V[1*5+4]=V[9]
    V[4, col:col+6] = V3[0, :]
    V[9, col:col+6] = V3[1, :]
    # W for C[4:5, 0:3]: k=4; i in 0,1,2. W[4*3+0]=W[12], W[13], W[14]
    W[12:15, col:col+6] = W3
    col += 6
    
    # Part 4: A[0:3, 2:5] (3x3), B[2:5, 0:2] (3x2), C[0:2, 0:3] (2x3)
    U[6:15, col:col+17] = U4
    # V for B[2:5, 0:2]: j in 2,3,4; k in 0,1.
    # V[2*5+0]=V[10], V[2*5+1]=V[11], V[3*5+0]=V[15], V[16], V[4*5+0]=V[20], V[21]
    # V4 is for 3x2 B, shape (6, 17). Rows: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    # Map to global: (0,0)->10, (0,1)->11, (1,0)->15, (1,1)->16, (2,0)->20, (2,1)->21
    V[10, col:col+17] = V4[0, :]
    V[11, col:col+17] = V4[1, :]
    V[15, col:col+17] = V4[2, :]
    V[16, col:col+17] = V4[3, :]
    V[20, col:col+17] = V4[4, :]
    V[21, col:col+17] = V4[5, :]
    # W for C[0:2, 0:3]: k in 0,1; i in 0,1,2. W[0:6]
    W[0:6, col:col+17] = W4
    col += 17
    
    # Part 5: A[0:3, 2:5] (3x3), B[2:5, 2:5] (3x3), C[2:5, 0:3] (3x3)
    U[6:15, col:col+27] = U5
    # V for B[2:5, 2:5]: j in 2,3,4; k in 2,3,4.
    # V[2*5+2]=V[12], V[13], V[14], V[3*5+2]=V[17], V[18], V[19], V[4*5+2]=V[22], V[23], V[24]
    # V5 is for 3x3 B, shape (9, 27). Rows: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    V[12, col:col+27] = V5[0, :]
    V[13, col:col+27] = V5[1, :]
    V[14, col:col+27] = V5[2, :]
    V[17, col:col+27] = V5[3, :]
    V[18, col:col+27] = V5[4, :]
    V[19, col:col+27] = V5[5, :]
    V[22, col:col+27] = V5[6, :]
    V[23, col:col+27] = V5[7, :]
    V[24, col:col+27] = V5[8, :]
    # W for C[2:5, 0:3]: k in 2,3,4; i in 0,1,2. W[6:15]
    W[6:15, col:col+27] = W5
    col += 27
    
    # Convert to lists of integers/halves
    U_list = [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
