import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_val, v_val, w_val, u_idx, v_idx, w_idx):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: A1 * B1,0 (cols 0,1 of C)
    # A1 is A[0:2, 0:2]
    # B1,0 is B[0:2, 0:2]
    # Strassen
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11.
    # This is a sum of products. We need to distribute.
    # The tensor decomposition is a sum of rank 1 terms.
    # Strassen's algorithm computes 7 intermediate products.
    # Each product is a scalar multiplication.
    # So we add 7 terms.
    
    # Term 1: M1
    # (A00 + A11) * (B00 + B11)
    # This expands to A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in tensor decomposition, we can have coefficients in U, V, W.
    # U[i*m+j] corresponds to A[i,j].
    # V[j*p+k] corresponds to B[j,k].
    # W[k*n+i] corresponds to C[i,k].
    
    # For M1:
    # We want to compute (A00+A11)(B00+B11).
    # We can set U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1.
    # Then the product is (A00+A11)(B00+B11).
    # This contributes to C.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M2 + M6
    # C10 = M3 + M2
    # C11 = M1 - M3 + M4
    
    # So M1 contributes +1 to C00, +1 to C01, +1 to C11? No, M1 is in C00, C01, C11?
    # C11 = M1 - M3 + M4. Yes.
    # C10 does NOT have M1.
    
    # So for M1:
    # U[0,0] += 1, U[1,1] += 1
    # V[0,0] += 1, V[1,1] += 1
    # W[0,0] += 1 (C00)
    # W[1,0] += 1 (C01) -- Wait, W index is k*n+i.
    # C[i,k]. W[k*n+i].
    # C00 -> k=0, i=0 -> W[0]
    # C01 -> k=1, i=0 -> W[2]
    # C10 -> k=0, i=1 -> W[1]
    # C11 -> k=1, i=1 -> W[3]
    
    # Let's define a helper to add a Strassen block for specific A sub-block and B sub-block.
    
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start):
        nonlocal col
        # A is 2x2 at (a_row_start, a_col_start)
        # B is 2x2 at (b_row_start, b_col_start)
        # Indices in U: (a_row_start + dr) * m + (a_col_start + dc)
        # Indices in V: (b_row_start + dr) * p + (b_col_start + dc)
        # Indices in W: (b_col_start + dc) * n + (a_row_start + dr)
        
        # M1 = (A00+A11)(B00+B11)
        U[a_row_start*m + a_col_start][col] += 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] += 1
        V[b_row_start*p + b_col_start][col] += 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] += 1
        W[b_col_start*n + a_row_start][col] += 1      # C00
        W[(b_col_start+1)*n + a_row_start][col] += 1  # C01
        W[(b_col_start+1)*n + (a_row_start+1)][col] += 1 # C11
        col += 1
        
        # M2 = (A10+A11)B00
        U[(a_row_start+1)*m + a_col_start][col] += 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] += 1
        V[b_row_start*p + b_col_start][col] += 1
        W[(b_col_start+1)*n + a_row_start][col] += -1 # C01
        W[b_col_start*n + (a_row_start+1)][col] += 1  # C10
        col += 1
        
        # M3 = A00(B10-B11)
        U[a_row_start*m + a_col_start][col] += 1
        V[(b_row_start+1)*p + b_col_start][col] += 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] += -1
        W[(b_col_start+1)*n + a_row_start][col] += 1  # C01
        W[b_col_start*n + (a_row_start+1)][col] += 1  # C10
        W[(b_col_start+1)*n + (a_row_start+1)][col] += -1 # C11
        col += 1
        
        # M4 = A11(B00-B01)
        U[(a_row_start+1)*m + (a_col_start+1)][col] += 1
        V[b_row_start*p + b_col_start][col] += 1
        V[b_row_start*p + (b_col_start+1)][col] += -1
        W[b_col_start*n + a_row_start][col] += 1      # C00
        W[(b_col_start+1)*n + (a_row_start+1)][col] += 1 # C11
        col += 1
        
        # M5 = (A00+A10)B11
        U[a_row_start*m + a_col_start][col] += 1
        U[(a_row_start+1)*m + a_col_start][col] += 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] += 1
        W[b_col_start*n + a_row_start][col] += -1     # C00
        col += 1
        
        # M6 = (A11-A00)(B00+B01)
        U[(a_row_start+1)*m + (a_col_start+1)][col] += 1
        U[a_row_start*m + a_col_start][col] += -1
        V[b_row_start*p + b_col_start][col] += 1
        V[b_row_start*p + (b_col_start+1)][col] += 1
        W[(b_col_start+1)*n + a_row_start][col] += 1  # C01
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_row_start*m + (a_col_start+1)][col] += 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] += -1
        V[(b_row_start+1)*p + b_col_start][col] += 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] += 1
        W[b_col_start*n + a_row_start][col] += 1      # C00
        W[(b_col_start+1)*n + (a_row_start+1)][col] += 1 # C11
        col += 1

    # Apply Strassen for B1,0 (rows 0,1 of B, cols 0,1)
    # A1 is rows 0,1 of A, cols 0,1.
    add_strassen_block(0, 0, 0, 0)
    
    # Apply Strassen for B1,1 (rows 0,1 of B, cols 2,3)
    add_strassen_block(0, 0, 0, 2)
    
    # Apply Strassen for B1,2 (rows 0,1 of B, cols 4,5)
    add_strassen_block(0, 0, 0, 4)
    
    # Part 2: A1 * B1,3 (col 6 of C)
    # A1 is 2x2 (rows 0,1, cols 0,1). B1,3 is 2x1 (rows 0,1, col 6).
    # C0,6 = A00*B0,6 + A01*B1,6
    # C1,6 = A10*B0,6 + A11*B1,6
    
    # Term 1: A00 * B0,6
    U[0*m + 0][col] += 1
    V[0*p + 6][col] += 1
    W[6*n + 0][col] += 1 # C0,6
    col += 1
    
    # Term 2: A01 * B1,6
    U[0*m + 1][col] += 1
    V[1*p + 6][col] += 1
    W[6*n + 0][col] += 1 # C0,6
    col += 1
    
    # Term 3: A10 * B0,6
    U[1*m + 0][col] += 1
    V[0*p + 6][col] += 1
    W[6*n + 1][col] += 1 # C1,6
    col += 1
    
    # Term 4: A11 * B1,6
    U[1*m + 1][col] += 1
    V[1*p + 6][col] += 1
    W[6*n + 1][col] += 1 # C1,6
    col += 1
    
    # Part 3: A2 * B2
    # A2 is 2x1 (rows 0,1, col 2). B2 is 1x7 (row 2, cols 0..6).
    # C0,k = A0,2 * B2,k
    # C1,k = A1,2 * B2,k
    
    for k in range(7):
        # Term: A0,2 * B2,k
        U[0*m + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # Term: A1,2 * B2,k
        U[1*m + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    if col != rank:
        # This should not happen
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

DIMENSIONS = [2, 3, 7]
