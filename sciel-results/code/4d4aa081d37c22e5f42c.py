import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices for the full 2x5x6 problem.
    # U shape: (2*5, rank) = (10, rank)
    # V shape: (5*6, rank) = (30, rank)
    # W shape: (6*2, rank) = (12, rank)
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_222(offset_u, offset_v, offset_w):
        """
        Adds a 2x2x2 Strassen decomposition.
        offset_u: index in U's row space (i*m + j) for the first row of the 2x2 A block.
                  The 2x2 A block occupies rows [offset_u, offset_u+1] and [offset_u+m, offset_u+m+1]?
                  No, U rows are i*m + j.
                  For a 2x2 block starting at row i, col j of A:
                  A[0,0] is at U[i*m + j]
                  A[0,1] is at U[i*m + j + 1]
                  A[1,0] is at U[(i+1)*m + j]
                  A[1,1] is at U[(i+1)*m + j + 1]
                  
        offset_v: index in V's row space (j*p + k) for the first row of the 2x2 B block.
                  B[0,0] is at V[j*p + k]
                  B[0,1] is at V[j*p + k + 1]
                  B[1,0] is at V[(j+1)*p + k]
                  B[1,1] is at V[(j+1)*p + k + 1]
                  
        offset_w: index in W's row space (k*n + i) for the first col of the 2x2 C block.
                  C[0,0] is at W[k*n + i]
                  C[0,1] is at W[k*n + i + 1]
                  C[1,0] is at W[(k+1)*n + i]
                  C[1,1] is at W[(k+1)*n + i + 1]
        """
        nonlocal rank
        nonlocal U_cols, V_cols, W_cols
        
        # Strassen coefficients for 2x2x2
        # U: 7 columns
        # V: 7 columns
        # W: 7 columns
        
        # A indices relative to offset_u
        a11_u = offset_u
        a12_u = offset_u + 1
        a21_u = offset_u + m
        a22_u = offset_u + m + 1
        
        # B indices relative to offset_v
        b11_v = offset_v
        b12_v = offset_v + 1
        b21_v = offset_v + p
        b22_v = offset_v + p + 1
        
        # C indices relative to offset_w
        c11_w = offset_w
        c12_w = offset_w + 1
        c21_w = offset_w + n
        c22_w = offset_w + n + 1
        
        # M1 = (A11+A22)(B11+B22)
        u1 = [0]*rank; v1 = [0]*rank; w1 = [0]*rank
        # We are appending, so we just define the new column
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        
        new_u[a11_u] = 1
        new_u[a22_u] = 1
        new_v[b11_v] = 1
        new_v[b22_v] = 1
        new_w[c11_w] = 1
        new_w[c12_w] = 1
        new_w[c21_w] = -1
        new_w[c22_w] = -1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M2 = (A21+A22)B22
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a21_u] = 1
        new_u[a22_u] = 1
        new_v[b22_v] = 1
        new_w[c21_w] = 1
        new_w[c22_w] = 1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M3 = A11(B12-B22)
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a11_u] = 1
        new_v[b12_v] = 1
        new_v[b22_v] = -1
        new_w[c12_w] = 1
        new_w[c22_w] = 1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M4 = A22(B21-B11)
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a22_u] = 1
        new_v[b21_v] = 1
        new_v[b11_v] = -1
        new_w[c11_w] = 1
        new_w[c21_w] = 1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M5 = (A11+A12)B11
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a11_u] = 1
        new_u[a12_u] = 1
        new_v[b11_v] = 1
        new_w[c11_w] = -1
        new_w[c12_w] = -1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M6 = (A21-A11)(B11+B12)
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a21_u] = 1
        new_u[a11_u] = -1
        new_v[b11_v] = 1
        new_v[b12_v] = 1
        new_w[c22_w] = 1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        # M7 = (A12-A22)(B21+B22)
        new_u = [0] * 10
        new_v = [0] * 30
        new_w = [0] * 12
        new_u[a12_u] = 1
        new_u[a22_u] = -1
        new_v[b21_v] = 1
        new_v[b22_v] = 1
        new_w[c11_w] = -1
        new_w[c12_w] = -1
        new_w[c21_w] = -1
        new_w[c22_w] = -1
        U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
        
        rank += 7

    def add_schoolbook_216(offset_u, offset_v, offset_w):
        """
        Adds a 2x1x6 Schoolbook decomposition.
        A is 2x1, B is 1x6, C is 2x6.
        offset_u: row of A (i*m + j)
        offset_v: row of B (j*p + k)
        offset_w: col of C (k*n + i)
        """
        nonlocal rank
        nonlocal U_cols, V_cols, W_cols
        
        a1_u = offset_u
        a2_u = offset_u + m
        
        b_v = offset_v
        
        # 6 terms, one for each column of B/C
        for k in range(6):
            new_u = [0] * 10
            new_v = [0] * 30
            new_w = [0] * 12
            
            new_u[a1_u] = 1
            new_u[a2_u] = 1
            
            new_v[b_v + k] = 1
            
            new_w[offset_w + k] = 1
            new_w[offset_w + k + n] = 1
            
            U_cols.append(new_u); V_cols.append(new_v); W_cols.append(new_w)
            rank += 1

    # Decomposition Plan:
    # A is 2x5. Split into A1 (2x2) and A2 (2x3).
    # B is 5x6. Split into B1 (2x6) and B2 (3x6).
    # C = A1*B1 + A2*B2.
    
    # A1 is rows 0-1, cols 0-1 of A.
    # A2 is rows 0-1, cols 2-4 of A.
    
    # B1 is rows 0-1, cols 0-5 of B.
    # B2 is rows 2-4, cols 0-5 of B.
    
    # Term 1: A1 * B1 (2x2x6)
    # Use 3 Strassen 2x2x2 blocks.
    # B1 is 2x6. Split into B1_1 (2x2), B1_2 (2x2), B1_3 (2x2).
    
    # B1_1: B rows 0-1, cols 0-1.
    # A1: A rows 0-1, cols 0-1.
    # C part: C rows 0-1, cols 0-1.
    add_strassen_222(offset_u=0, offset_v=0, offset_w=0)
    
    # B1_2: B rows 0-1, cols 2-3.
    # A1: A rows 0-1, cols 0-1.
    # C part: C rows 0-1, cols 2-3.
    add_strassen_222(offset_u=0, offset_v=2, offset_w=2)
    
    # B1_3: B rows 0-1, cols 4-5.
    # A1: A rows 0-1, cols 0-1.
    # C part: C rows 0-1, cols 4-5.
    add_strassen_222(offset_u=0, offset_v=4, offset_w=4)
    
    # Term 2: A2 * B2 (2x3x6)
    # A2 is 2x3. Split into A2_1 (2x1) and A2_2 (2x2).
    # B2 is 3x6. Split into B2_1 (1x6) and B2_2 (2x6).
    # A2*B2 = A2_1*B2_1 + A2_2*B2_2.
    
    # A2_1: A rows 0-1, col 2.
    # B2_1: B row 2, cols 0-5.
    # C part: C rows 0-1, cols 0-5.
    add_schoolbook_216(offset_u=2, offset_v=12, offset_w=0)
    # Note: B row 2 starts at index 2*6 = 12 in V.
    
    # A2_2: A rows 0-1, cols 3-4.
    # B2_2: B rows 3-4, cols 0-5.
    # C part: C rows 0-1, cols 0-5.
    # Use 3 Strassen 2x2x2 blocks for 2x2x6.
    
    # B2_2_1: B rows 3-4, cols 0-1.
    add_strassen_222(offset_u=3, offset_v=18, offset_w=0)
    # Note: A col 3 starts at index 3 in U row 0? No.
    # U index is i*m + j.
    # A2_2 is rows 0-1, cols 3-4.
    # A[0,3] is at U[0*5+3] = U[3].
    # A[0,4] is at U[4].
    # A[1,3] is at U[1*5+3] = U[8].
    # A[1,4] is at U[9].
    # So offset_u should be 3.
    
    # B2_2_1: B rows 3-4, cols 0-1.
    # B[3,0] is at V[3*6+0] = V[18].
    # So offset_v is 18.
    
    # C part: C rows 0-1, cols 0-1.
    # C[0,0] is at W[0*2+0] = W[0].
    # So offset_w is 0.
    
    # B2_2_2: B rows 3-4, cols 2-3.
    add_strassen_222(offset_u=3, offset_v=20, offset_w=2)
    
    # B2_2_3: B rows 3-4, cols 4-5.
    add_strassen_222(offset_u=3, offset_v=22, offset_w=4)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
