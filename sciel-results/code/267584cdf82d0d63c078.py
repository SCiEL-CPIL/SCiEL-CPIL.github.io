"""Strassen-based decomposition for 2x4x6 matrix multiplication.

The 2x4x6 tensor splits into 6 independent 2x2x2 sub-tensors:
  A is 2x4 -> two row-blocks (rows 0-1, cols 0-1) and (rows 0-1, cols 2-3)
  B is 4x6 -> three column-blocks (cols 0-1, 2-3, 4-5)
Each 2x2x2 sub-tensor uses Strassen's rank-7 decomposition.
Total rank = 6 * 7 = 42.

Index conventions:
  U[i*m + j, c]  where i in [0,n), j in [0,m), c in [0,rank)
  V[j*p + k, c]  where j in [0,m), k in [0,p), c in [0,rank)
  W[k*n + i, c]  where k in [0,p), i in [0,n), c in [0,rank)
"""

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U, V, W) for the 2x2x2 Strassen algorithm, each shape (4,7).
    
    U indexed by (i,j) -> i*2+j
    V indexed by (j,k) -> j*2+k
    W indexed by (k,i) -> k*2+i
    
    Strassen's 7 products:
      M0 = (A00+A11)(B00+B11)
      M1 = (A10+A11) B00
      M2 = A00 (B10-B00)
      M3 = A11 (B11-B01)
      M4 = (A00+A01) B11
      M5 = (A10-A00)(B00+B01)
      M6 = (A01-A11)(B10+B11)
      
      C00 = M0+M3-M4-M6
      C01 = M2+M4
      C10 = M1+M5
      C11 = M0-M1+M2+M6
    """
    n = m = p = 2
    r = 7
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # M0 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 gets +M0, C11 gets +M0
    # Actually W is per-output. Let me be more careful.
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M6
    
    # Each M_c contributes to specific C entries with specific signs.
    # U[c] tells which (i,j) pairs enter M_c (with their coefficients in A)
    # V[c] tells which (j,k) pairs enter M_c (with their coefficients in B)
    # W[c] tells which (k,i) pairs get M_c (with their coefficients in C)
    
    # M0: A part = A00+A11, B part = B00+B11
    # C part: C00 += M0, C11 += M0
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # M1: A part = A10+A11, B part = B00
    # C part: C10 += M1, C11 -= M1
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # M2: A part = A00, B part = B10-B00
    # C part: C01 += M2, C11 += M2
    U[0*2+0][2] = 1
    V[1*2+0][2] = 1; V[0*2+0][2] = -1
    W[0*2+0][2] = 1; W[1*2+1][2] = 1
    
    # M3: A part = A11, B part = B11-B01
    # C part: C00 += M3, C11: not in formula... wait
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M6
    # So M3 only appears in C00.
    U[1*2+1][3] = 1
    V[1*2+1][3] = 1; V[0*2+1][3] = -1
    W[0*2+0][3] = 1
    
    # M4: A part = A00+A01, B part = B11
    # C part: C00 -= M4, C01 += M4
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[0*2+1][4] = 1
    
    # M5: A part = A10-A00, B part = B00+B01
    # C part: C10 += M5
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[0*2+1][5] = 1
    
    # M6: A part = A01-A11, B part = B10+B11
    # C part: C00 -= M6, C11 += M6
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = -1; W[1*2+1][6] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Decompose into 6 sub-problems: 2 row-blocks x 3 col-blocks
    # A is n x m = 2 x 4: split m into blocks of 2: [0,2), [2,4)
    # B is m x p = 4 x 6: split p into blocks of 2: [0,2), [2,4), [4,6)
    # But we also need to split A's rows? n=2, so no row splitting needed.
    # Actually for 2x2x2 Strassen, we need A to be 2x2 and B to be 2x2.
    # A is 2x4: split columns into two 2x2 blocks.
    # B is 4x6: split into (2x2) blocks. Rows of B correspond to columns of A.
    #   B rows 0-1 pair with A cols 0-1 (block 0)
    #   B rows 2-3 pair with A cols 2-3 (block 1)
    #   B cols split into 3 blocks of 2.
    
    # So we have 2 x 3 = 6 sub-problems, each is 2x2x2.
    # Sub-problem (ai, bi): ai in {0,1}, bi in {0,1,2}
    #   A sub-matrix: rows 0-1 (all n=2 rows), cols ai*2 to ai*2+1
    #   B sub-matrix: rows ai*2 to ai*2+1, cols bi*2 to bi*2+1
    #   C sub-matrix: rows bi*2 to bi*2+1, cols 0-1 (all n=2 cols)
    
    rank = 6 * 7  # 42
    U = [[0]*rank for _ in range(n*m)]  # 8 x 42
    V = [[0]*rank for _ in range(m*p)]  # 24 x 42
    W = [[0]*rank for _ in range(p*n)]  # 12 x 42
    
    U_s, V_s, W_s = _strassen_factors()
    
    col_offset = 0
    for ai in range(2):      # A column block index (0 or 1)
        for bi in range(3):  # B column block index (0, 1, or 2)
            # For this sub-problem:
            # A local (i,j) where i in {0,1}, j in {0,1}
            #   global A index: i*m + (ai*2 + j) = i*4 + ai*2 + j
            # B local (j,k) where j in {0,1}, k in {0,1}
            #   global B index: (ai*2 + j)*p + (bi*2 + k) = (ai*2+j)*6 + bi*2 + k
            # C local (k,i) where k in {0,1}, i in {0,1}
            #   global C index: (bi*2 + k)*n + i = (bi*2+k)*2 + i
            
            for c in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        val = U_s[i*2+j][c]
                        if val != 0:
                            U[i*4 + ai*2 + j][col_offset + c] = val
                # V
                for j in range(2):
                    for k in range(2):
                        val = V_s[j*2+k][c]
                        if val != 0:
                            V[(ai*2+j)*6 + bi*2 + k][col_offset + c] = val
                # W
                for k in range(2):
                    for i in range(2):
                        val = W_s[k*2+i][c]
                        if val != 0:
                            W[(bi*2+k)*2 + i][col_offset + c] = val
            col_offset += 7
    
    return {"U": U, "V": V, "W": W}
