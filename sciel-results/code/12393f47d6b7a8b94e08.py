"""Matrix multiplication 4x5 by 5x5 - Strassen-block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 2.0

def _strassen_2x2x2(r, iA, iB, iC, U, V, W, start_col):
    """Add a 2x2 Strassen decomposition.
    
    A is 2x2 at rows iA=[r0,r1], cols [c0,c1]
    B is 2x2 at rows [d0,d1], cols [e0,e1]
    C is 2x2 at rows [f0,f1], cols [g0,g1]
    
    Strassen factors:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    U index: r*m + j  where r is row of A, j is col of A
    V index: j*p + k  where j is row of B, k is col of B
    W index: k*n + i  where k is row of C, i is col of C
    """
    m, p, n = DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[0]
    col = start_col
    
    # M1 = (A11+A22)(B11+B22)
    # A11: row iA[0], col iA[0] -> U[iA[0]*m + iA[0]]
    # A22: row iA[1], col iA[1] -> U[iA[1]*m + iA[1]]
    # B11: row iB[0], col iB[0] -> V[iB[0]*p + iB[0]]
    # B22: row iB[1], col iB[1] -> V[iB[1]*p + iB[1]]
    # C11: row iC[0], col iC[0] -> W[iC[0]*n + iC[0]]
    # C22: row iC[1], col iC[1] -> W[iC[1]*n + iC[1]]
    
    # M1: A11*B11 + A22*B22 contributes to C11 and C22
    # But Strassen is a bilinear form, so each M is one term
    # M1 = (A11+A22)(B11+B22) expands to A11*B11 + A11*B22 + A22*B11 + A22*B22
    # This is NOT a single rank-1 term. Strassen uses 7 bilinear terms.
    # Each bilinear term is: sum of products of linear combinations.
    
    # Actually, each M_i is a single bilinear form:
    # M1 = (A11+A22)(B11+B22) means:
    #   The bilinear term is (U_A11 + U_A22) * (V_B11 + V_B22)
    #   which contributes to C with W coefficients.
    
    # For the tensor decomposition, each rank-1 component is:
    #   u_vec * v_vec * w_vec
    # where u_vec has entries for the A indices, v_vec for B, w_vec for C.
    
    # M1: u = e_A11 + e_A22, v = e_B11 + e_B22, w = e_C11 + e_C22
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,k} (in appropriate indexing)
    # For 2x2x2: T[i,j,k] = 1 if the indices satisfy the multiplication pattern.
    
    # Actually for C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]:
    # T[i,j,k] = delta (the tensor has T[i,j,k]=1 for valid (i,j,k))
    # In rank-1 decomposition: T = sum_r u_r[i] * v_r[j] * w_r[k]
    
    # For schoolbook: each (i,j,k) gives u=e_i, v=e_j, w=e_k
    # For Strassen on 2x2x2:
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # Where:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # Each M_r is a product of a linear form in A and a linear form in B.
    # So u_r is the vector of coefficients for A, v_r for B, w_r for C.
    
    # M1: u=(1,0,0,1) [A00,A01,A10,A11], v=(1,0,0,1) [B00,B01,B10,B11], 
    #      w=(1,0,1,0) [C00,C01,C10,C11] -> contributes +1 to C00 and +1 to C11
    # Wait: C00 gets +M1, C11 gets +M1. So w=(1,0,0,1) in (C00,C01,C10,C11) ordering.
    
    # Let me use row-major: A index = i*m+j = i*2+j, B index = j*2+k = j*2+k, C index = k*2+i = k*2+i
    # A: [A00, A01, A10, A11] indices 0,1,2,3
    # B: [B00, B01, B10, B11] indices 0,1,2,3
    # C: [C00, C01, C10, C11] indices 0,1,2,3 (k*2+i: C00=0*2+0=0, C01=0*2+1=1, C10=1*2+0=2, C11=1*2+1=3)
    
    # M1: u_A = [1,0,0,1] (A00+A11), v_B = [1,0,0,1] (B00+B11), w_C = [1,0,0,1] (C00+C11)
    # M2: u_A = [0,0,1,1] (A10+A11), v_B = [1,0,0,0] (B00), w_C = [0,0,1,0] (C10)
    #     C10 gets +M2, C11 gets -M2. So w_C = [0,0,1,-1]
    # M3: u_A = [1,0,0,0] (A00), v_B = [0,0,1,-1] (B10-B11), w_C = [0,1,0,1] (C01+C11)
    # M4: u_A = [0,0,0,1] (A11), v_B = [0,1,-1,0] (B01-B00), w_C = [1,0,1,0] (C00+C10)
    # M5: u_A = [1,1,0,0] (A00+A01), v_B = [0,0,0,1] (B11), w_C = [-1,1,0,0] (-C00+C01)
    # M6: u_A = [0,0,1,-1] (A10-A00), v_B = [1,1,0,0] (B00+B01), w_C = [0,0,0,1] (C11)
    # M7: u_A = [0,1,0,-1] (A01-A11), v_B = [0,0,1,1] (B10+B11), w_C = [1,0,0,1] (C00+C11)
    
    # Now map to global indices:
    # A global index: iA_global[i] = (r0+i)*m + (c0+j) for local (i,j)
    # B global index: iB_global[j] = (d0+j)*p + (e0+k) for local (j,k)
    # C global index: iC_global[k] = (f0+k)*n + (g0+i) for local (k,i)
    
    # Local A indices (i,j): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local B indices (j,k): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local C indices (k,i): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Global A for local (i,j): row = r0+i, col = c0+j, index = (r0+i)*m + (c0+j)
    # Global B for local (j,k): row = d0+j, col = e0+k, index = (d0+j)*p + (e0+k)
    # Global C for local (k,i): row = f0+k, col = g0+i, index = (f0+k)*n + (g0+i)
    
    r0, r1 = iA[0], iA[1]
    c0, c1 = iA[0], iA[1]  # A is square 2x2, so cols same as rows in local
    d0, d1 = iB[0], iB[1]
    e0, e1 = iB[0], iB[1]
    f0, f1 = iC[0], iC[1]
    g0, g1 = iC[0], iC[1]
    
    # Global indices
    A00 = r0*m + c0
    A01 = r0*m + c1
    A10 = r1*m + c0
    A11 = r1*m + c1
    
    B00 = d0*p + e0
    B01 = d0*p + e1
    B10 = d1*p + e0
    B11 = d1*p + e1
    
    C00 = f0*n + g0
    C01 = f0*n + g1
    C10 = f1*n + g0
    C11 = f1*n + g1
    
    # M1: u=[A00:A11], v=[B00:B11], w=[C00,C11]
    col += 1
    U[A00][col] = 1
    U[A11][col] = 1
    V[B00][col] = 1
    V[B11][col] = 1
    W[C00][col] = 1
    W[C11][col] = 1
    
    # M2: u=[A10,A11], v=[B00], w=[C10:+1, C11:-1]
    col += 1
    U[A10][col] = 1
    U[A11][col] = 1
    V[B00][col] = 1
    W[C10][col] = 1
    W[C11][col] = -1
    
    # M3: u=[A00], v=[B10-B11], w=[C01:+1, C11:+1]
    col += 1
    U[A00][col] = 1
    V[B10][col] = 1
    V[B11][col] = -1
    W[C01][col] = 1
    W[C11][col] = 1
    
    # M4: u=[A11], v=[B01-B00], w=[C00:+1, C10:+1]
    col += 1
    U[A11][col] = 1
    V[B01][col] = 1
    V[B00][col] = -1
    W[C00][col] = 1
    W[C10][col] = 1
    
    # M5: u=[A00+A01], v=[B11], w=[C00:-1, C01:+1]
    col += 1
    U[A00][col] = 1
    U[A01][col] = 1
    V[B11][col] = 1
    W[C00][col] = -1
    W[C01][col] = 1
    
    # M6: u=[A10-A00], v=[B00+B01], w=[C11:+1]
    col += 1
    U[A10][col] = 1
    U[A00][col] = -1
    V[B00][col] = 1
    V[B01][col] = 1
    W[C11][col] = 1
    
    # M7: u=[A01-A11], v=[B10+B11], w=[C00:+1, C11:+1]
    col += 1
    U[A01][col] = 1
    U[A11][col] = -1
    V[B10][col] = 1
    V[B11][col] = 1
    W[C00][col] = 1
    W[C11][col] = 1
    
    return col  # new column count (7 terms added)

def _schoolbook_block(r, rows_A, cols_A, rows_B, cols_B, rows_C, cols_C, U, V, W, start_col):
    """Add schoolbook decomposition for a rectangular block.
    
    A[rows_A, cols_A] * B[rows_B, cols_B] -> C[rows_C, cols_C]
    where len(cols_A) == len(rows_B) = m_block
    """
    m, p, n = DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[0]
    col = start_col
    for j in range(len(cols_A)):
        a_col = cols_A[j]
        b_row = rows_B[j]
        for i in rows_A:
            for k in cols_C:
                col += 1
                U[i*m + a_col][col] = 1
                V[b_row*p + k][col] = 1
                W[k*n + i][col] = 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    U = [[0] * (n*m*p) for _ in range(n*m)]  # start with schoolbook size, we'll use a subset
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    # We'll build the decomposition and track columns used
    # Plan: partition into 2x2 Strassen blocks and 1-wide schoolbook blocks
    
    # A is 4x5, B is 5x5, C is 4x5
    # Split A rows into [0:2] and [2:4]
    # Split A cols into [0:2], [2:3], [3:5]
    # Split B rows (matching A cols) into [0:2], [2:3], [3:5]
    # Split B cols into [0:2], [2:3], [3:5]
    # Split C rows into [0:2], [2:4]
    # Split C cols into [0:2], [2:3], [3:5]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen 2x2x2
    col = _strassen_2x2x2(n, [0,1], [0,1], [0,1], U, V, W, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:3] -> C[0:2, 2:3] : 2x2x2x1 -> schoolbook (2*2*1=4)
    col = _schoolbook_block(n, [0,1], [0,1], [0,1], [2], [0,1], [2], U, V, W, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 3:5] -> C[0:2, 3:5] : 2x2x2x2 -> Strassen
    col = _strassen_2x2x2(n, [0,1], [0,1], [0,1], U, V, W, col)
    # Wait, B cols are 3:5 which is 2 wide, C cols are 3:5 which is 2 wide.
    # But B rows are 0:2, A cols are 0:2. So it's 2x2 * 2x2 -> 2x2. Yes Strassen.
    # But I need to map the global indices correctly. The function uses iA, iB, iC as
    # the global row/col indices. Let me fix the function to take explicit index lists.
    
    # Let me redo this more carefully.

    # Reset
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    
    def strassen(rowsA, colsA, rowsB, colsB, rowsC, colsC, U, V, W, col):
        """Strassen for 2x2 blocks with explicit global indices."""
        m, p, n = DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[0]
        # Local to global
        A00 = rowsA[0]*m + colsA[0]
        A01 = rowsA[0]*m + colsA[1]
        A10 = rowsA[1]*m + colsA[0]
        A11 = rowsA[1]*m + colsA[1]
        B00 = rowsB[0]*p + colsB[0]
        B01 = rowsB[0]*p + colsB[1]
        B10 = rowsB[1]*p + colsB[0]
        B11 = rowsB[1]*p + colsB[1]
        C00 = rowsC[0]*n + colsC[0]
        C01 = rowsC[0]*n + colsC[1]
        C10 = rowsC[1]*n + colsC[0]
        C11 = rowsC[1]*n + colsC[1]
        
        # M1
        col += 1
        U[A00][col]=1; U[A11][col]=1; V[B00][col]=1; V[B11][col]=1; W[C00][col]=1; W[C11][col]=1
        # M2
        col += 1
        U[A10][col]=1; U[A11][col]=1; V[B00][col]=1; W[C10][col]=1; W[C11][col]=-1
        # M3
        col += 1
        U[A00][col]=1; V[B10][col]=1; V[B11][col]=-1; W[C01][col]=1; W[C11][col]=1
        # M4
        col += 1
        U[A11][col]=1; V[B01][col]=1; V[B00][col]=-1; W[C00][col]=1; W[C10][col]=1
        # M5
        col += 1
        U[A00][col]=1; U[A01][col]=1; V[B11][col]=1; W[C00][col]=-1; W[C01][col]=1
        # M6
        col += 1
        U[A10][col]=1; U[A00][col]=-1; V[B00][col]=1; V[B01][col]=1; W[C11][col]=1
        # M7
        col += 1
        U[A01][col]=1; U[A11][col]=-1; V[B10][col]=1; V[B11][col]=1; W[C00][col]=1; W[C11][col]=1
        return col
    
    def schoolbook(rowsA, colsA, rowsB, colsB, rowsC, colsC, U, V, W, col):
        """Schoolbook for blocks where at least one dimension is 1."""
        m, p, n = DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[0]
        for j_idx in range(len(colsA)):
            a_col = colsA[j_idx]
            b_row = rowsB[j_idx]
            for i in rowsA:
                for k in colsC:
                    col += 1
                    U[i*m + a_col][col] = 1
                    V[b_row*p + k][col] = 1
                    W[k*n + i][col] = 1
        return col
    
    # Now build the decomposition:
    # A rows: [0,1] (top), [2,3] (bottom)
    # A cols / B rows: [0,1] (left), [2] (mid), [3,4] (right)
    # B cols / C cols: [0,1] (left), [2] (mid), [3,4] (right)
    # C rows: [0,1] (top), [2,3] (bottom)
    
    # Top-left: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] : 2x2x2 Strassen
    col = strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], U,V,W, col)  # +7
    
    # Top-mid: A[0:2,0:2] * B[0:2,2] -> C[0:2,2] : 2x2x1x1 -> schoolbook 4
    col = schoolbook([0,1],[0,1],[0,1],[2],[0,1],[2], U,V,W, col)  # +4
    
    # Top-right: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5] : 2x2x2x2 Strassen
    col = strassen([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], U,V,W, col)  # +7
    
    # Bot-left: A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2] : 2x2x2x2 Strassen
    col = strassen([2,3],[0,1],[0,1],[0,1],[2,3],[0,1], U,V,W, col)  # +7
    
    # Bot-mid: A[2:4,0:2] * B[0:2,2] -> C[2:4,2] : 2x2x1x1 -> schoolbook 4
    col = schoolbook([2,3],[0,1],[0,1],[2],[2,3],[2], U,V,W, col)  # +4
    
    # Bot-right: A[2:4,0:2] * B[0:2,3:5] -> C[2:4,3:5] : 2x2x2x2 Strassen
    col = strassen([2,3],[0,1],[0,1],[3,4],[2,3],[3,4], U,V,W, col)  # +7
    
    # Top-mid2: A[0:2,2] * B[2,0:2] -> C[0:2,0:2] : 2x1x2x2 -> schoolbook 4
    col = schoolbook([0,1],[2],[2],[0,1],[0,1],[0,1], U,V,W, col)  # +4
    
    # Top-mid3: A[0:2,2] * B[2,2] -> C[0:2,2] : 2x1x1x1 -> schoolbook 2
    col = schoolbook([0,1],[2],[2],[2],[0,1],[2], U,V,W, col)  # +2
    
    # Top-mid4: A[0:2,2] * B[2,3:5] -> C[0:2,3:5] : 2x1x2x2 -> schoolbook 4
    col = schoolbook([0,1],[2],[2],[3,4],[0,1],[3,4], U,V,W, col)  # +4
    
    # Bot-mid5: A[2:4,2] * B[2,0:2] -> C[2:4,0:2] : 2x1x2x2 -> schoolbook 4
    col = schoolbook([2,3],[2],[2],[0,1],[2,3],[0,1], U,V,W, col)  # +4
    
    # Bot-mid6: A[2:4,2] * B[2,2] -> C[2:4,2] : 2x1x1x1 -> schoolbook 2
    col = schoolbook([2,3],[2],[2],[2],[2,3],[2], U,V,W, col)  # +2
    
    # Bot-mid7: A[2:4,2] * B[2,3:5] -> C[2:4,3:5] : 2x1x2x2 -> schoolbook 4
    col = schoolbook([2,3],[2],[2],[3,4],[2,3],[3,4], U,V,W, col)  # +4
    
    # Top-right2: A[0:2,3:5] * B[3:5,0:2] -> C[0:2,0:2] : 2x2x2x2 Strassen
    col = strassen([0,1],[3,4],[3,4],[0,1],[0,1],[0,1], U,V,W, col)  # +7
    
    # Top-right3: A[0:2,3:5] * B[3:5,2] -> C[0:2,2] : 2x2x1x1 -> schoolbook 4
    col = schoolbook([0,1],[3,4],[3,4],[2],[0,1],[2], U,V,W, col)  # +4
    
    # Top-right4: A[0:2,3:5] * B[3:5,3:5] -> C[0:2,3:5] : 2x2x2x2 Strassen
    col = strassen([0,1],[3,4],[3,4],[3,4],[0,1],[3,4], U,V,W, col)  # +7
    
    # Bot-right5: A[2:4,3:5] * B[3:5,0:2] -> C[2:4,0:2] : 2x2x2x2 Strassen
    col = strassen([2,3],[3,4],[3,4],[0,1],[2,3],[0,1], U,V,W, col)  # +7
    
    # Bot-right6: A[2:4,3:5] * B[3:5,2] -> C[2:4,2] : 2x2x1x1 -> schoolbook 4
    col = schoolbook([2,3],[3,4],[3,4],[2],[2,3],[2], U,V,W, col)  # +4
    
    # Bot-right7: A[2:4,3:5] * B[3:5,3:5] -> C[2:4,3:5] : 2x2x2x2 Strassen
    col = strassen([2,3],[3,4],[3,4],[3,4],[2,3],[3,4], U,V,W, col)  # +7
    
    # Total: 7*6 + 4*6 + 2*2 = 42 + 24 + 4 = 70
    
    # Trim to actual rank
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
