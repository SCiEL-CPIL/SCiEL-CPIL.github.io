import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Total Rank
    # Block 1 (2x2x2 Strassen): 7
    # Block 2 (2x2x2 Strassen): 7
    # Block 3 (1x2x2 Schoolbook): 2
    # Block 4 (1x2x2 Schoolbook): 2
    # Block 5 (1x1x2 Schoolbook): 2
    # Block 6 (1x1x2 Schoolbook): 2
    # Total R = 22
    
    R = 22
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Helper to add a block decomposition
    # a_offset, b_offset, c_offset are starting indices for rows/cols
    # a_rows, b_rows, c_rows are dimensions
    
    def add_strassen_block(u_start, v_start, w_start, rank_offset):
        # u_start: index of A[0,0] in flattened U (i*m + j)
        # v_start: index of B[0,0] in flattened V (j*p + k)
        # w_start: index of C[0,0] in flattened W (k*n + i)
        
        # Strassen factors for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in local 2x2 block:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Global indices:
        # U[ i*m + j ]
        # V[ j*p + k ]
        # W[ k*n + i ]
        
        # M1
        r = rank_offset
        # A part: A11+A22
        U[u_start, r] = 1 # A11
        U[u_start + 1, r] = 1 # A22 (i=1, j=1)
        # B part: B11+B22
        V[v_start, r] = 1 # B11
        V[v_start + p, r] = 1 # B22 (j=1, k=1)
        # C part: C11, C22
        W[w_start, r] = 1 # C11 (k=0, i=0)
        W[w_start + n, r] = 1 # C22 (k=1, i=1)
        
        # M2
        r = rank_offset + 1
        # A part: A21+A22
        U[u_start + 1, r] = 1 # A21 (i=1, j=0)
        U[u_start + 2, r] = 1 # A22 (i=1, j=1)
        # B part: B11
        V[v_start, r] = 1
        # C part: C21, C22
        W[w_start + 1, r] = 1 # C21 (k=0, i=1)
        W[w_start + n, r] = 1 # C22 (k=1, i=1)
        
        # M3
        r = rank_offset + 2
        # A part: A11
        U[u_start, r] = 1
        # B part: B21-B22
        V[v_start + p, r] = 1 # B21 (j=1, k=0)
        V[v_start + p + 1, r] = -1 # B22 (j=1, k=1)
        # C part: C12, C22
        W[w_start + 1, r] = 1 # C12 (k=0, i=1) -> Wait, C12 is k=1?
        # C index: k*n + i.
        # C11: k=0, i=0. C12: k=1, i=0.
        # C21: k=0, i=1. C22: k=1, i=1.
        # So C12 is w_start + 1. C22 is w_start + n.
        W[w_start + 1, r] = 1
        W[w_start + n, r] = 1
        
        # M4
        r = rank_offset + 3
        # A part: A22
        U[u_start + 2, r] = 1
        # B part: B12-B11
        V[v_start + 1, r] = 1 # B12 (j=0, k=1)
        V[v_start, r] = -1 # B11
        # C part: C11, C21
        W[w_start, r] = 1
        W[w_start + 1, r] = 1
        
        # M5
        r = rank_offset + 4
        # A part: A11+A12
        U[u_start, r] = 1
        U[u_start + 1, r] = 1 # A12 (i=0, j=1)
        # B part: B22
        V[v_start + p, r] = 1
        # C part: C11, C12
        W[w_start, r] = -1
        W[w_start + 1, r] = 1
        
        # M6
        r = rank_offset + 5
        # A part: A21-A11
        U[u_start + 1, r] = 1
        U[u_start, r] = -1
        # B part: B11+B12
        V[v_start, r] = 1
        V[v_start + 1, r] = 1
        # C part: C22
        W[w_start + n, r] = 1
        
        # M7
        r = rank_offset + 6
        # A part: A12-A22
        U[u_start + 1, r] = 1
        U[u_start + 2, r] = -1
        # B part: B21+B22
        V[v_start + p, r] = 1
        V[v_start + p + 1, r] = 1
        # C part: C11, C12
        W[w_start, r] = 1
        W[w_start + 1, r] = 1

    def add_schoolbook_1x2x2_block(u_start, v_start, w_start, rank_offset):
        # A is 1x2 (row i, cols j, j+1)
        # B is 2x2 (rows j, j+1; cols k, k+1)
        # C is 1x2 (row k, cols i, i+1)
        # Wait, dimensions:
        # A: 1 row, 2 cols.
        # B: 2 rows, 2 cols.
        # C: 1 row, 2 cols.
        # Multiplication: C[0, 0] = A[0, 0]B[0, 0] + A[0, 1]B[1, 0]
        # C[0, 1] = A[0, 0]B[0, 1] + A[0, 1]B[1, 1]
        
        # Rank 2.
        # r=0: A[0,0] * B[0, :] * C[:, 0] ?
        # Term 1: A[0,0] B[0,0] contributes to C[0,0]
        # Term 2: A[0,1] B[1,0] contributes to C[0,0]
        
        # Factor 1 (rank_offset):
        # U: A[0,0] = 1
        # V: B[0,0]=1, B[0,1]=1
        # W: C[0,0]=1
        r = rank_offset
        U[u_start, r] = 1
        V[v_start, r] = 1
        V[v_start + 1, r] = 1
        W[w_start, r] = 1
        
        # Factor 2 (rank_offset+1):
        # U: A[0,1] = 1
        # V: B[1,0]=1, B[1,1]=1
        # W: C[0,0]=1
        r = rank_offset + 1
        U[u_start + 1, r] = 1 # A[0,1] is at col 1 of A, so index +1
        V[v_start + p, r] = 1 # B[1,0] is at row 1 of B, so index +p
        V[v_start + p + 1, r] = 1
        W[w_start, r] = 1

    def add_schoolbook_1x1x2_block(u_start, v_start, w_start, rank_offset):
        # A is 1x1
        # B is 1x2
        # C is 1x2
        # C[0,0] = A[0,0] B[0,0]
        # C[0,1] = A[0,0] B[0,1]
        
        # Rank 2.
        # r=0: A[0,0]=1, B[0,0]=1, C[0,0]=1
        # r=1: A[0,0]=1, B[0,1]=1, C[0,1]=1
        
        r = rank_offset
        U[u_start, r] = 1
        V[v_start, r] = 1
        W[w_start, r] = 1
        
        r = rank_offset + 1
        U[u_start, r] = 1
        V[v_start + 1, r] = 1
        W[w_start + 1, r] = 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # u_start = 0*3+0 = 0
    # v_start = 0*4+0 = 0
    # w_start = 0*3+0 = 0
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # u_start = 0
    # v_start = 0*4+2 = 2
    # w_start = 2*3+0 = 6
    add_strassen_block(0, 2, 6, 7)
    
    # Block 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # A is 1x2 (row 2, cols 0,1). u_start = 2*3+0 = 6.
    # B is 2x2 (rows 0,1; cols 0,1). v_start = 0*4+0 = 0.
    # C is 1x2 (row 2, cols 0,1). w_start = 0*3+2 = 2.
    add_schoolbook_1x2x2_block(6, 0, 2, 14)
    
    # Block 4: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # u_start = 6
    # v_start = 2
    # w_start = 2*3+2 = 8
    add_schoolbook_1x2x2_block(6, 2, 8, 16)
    
    # Block 5: A[2:3, 2:3] * B[2:3, 0:2] -> C[2:3, 0:2]
    # A is 1x1 (row 2, col 2). u_start = 2*3+2 = 8.
    # B is 1x2 (row 2, cols 0,1). v_start = 2*4+0 = 8.
    # C is 1x2 (row 2, cols 0,1). w_start = 2.
    add_schoolbook_1x1x2_block(8, 8, 2, 18)
    
    # Block 6: A[2:3, 2:3] * B[2:3, 2:4] -> C[2:3, 2:4]
    # u_start = 8
    # v_start = 2*4+2 = 10
    # w_start = 2*3+2 = 8
    add_schoolbook_1x1x2_block(8, 10, 8, 20)

    # Convert to list of lists of integers/half-integers
    # All entries are integers in this construction.
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
