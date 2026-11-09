import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(col_offset_u, col_offset_v, col_offset_w, col_offset_u_row, col_offset_v_row, col_offset_w_row):
    """
    Return 7 Strassen terms for 2x2 x 2x2 multiplication.
    col_offset_u: offset in U rows (which 2 rows of the 2x2 block)
    col_offset_v: offset in V rows
    col_offset_w: offset in W rows
    col_offset_u_row: which row block (0 or 1) for U
    col_offset_v_row: which row block for V  
    col_offset_w_row: which row block for W
    Returns list of 7 (u_idx, v_idx, w_idx, coeff) tuples
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our tensor decomposition, each "multiplication" M corresponds to a rank-1 term
    # where U picks out a linear combo of A entries, V picks out a linear combo of B entries,
    # and W distributes the product to C entries.
    
    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # In U row indexing: row 0 = A11, row 1 = A12, row 2 = A21, row 3 = A22
    # B indices: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # In V row indexing: row 0 = B11, row 1 = B12, row 2 = B21, row 3 = B22
    # C indices: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    # In W row indexing: row 0 = C11, row 1 = C12, row 2 = C21, row 3 = C22
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11 + A22 -> rows [1, 0, 0, 1]
    # V: B11 + B22 -> rows [1, 0, 0, 1]
    # W: contributes to C11 (+1), C22 (+1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # U: A21 + A22 -> rows [0, 0, 1, 1]
    # V: B11 -> rows [1, 0, 0, 0]
    # W: contributes to C21 (+1), C22 (-1)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = A11(B21-B22)
    # U: A11 -> rows [1, 0, 0, 0]
    # V: B21 - B22 -> rows [0, 0, 1, -1]
    # W: contributes to C12 (+1), C22 (+1)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # M4 = A22(B12-B11)
    # U: A22 -> rows [0, 0, 0, 1]
    # V: B12 - B11 -> rows [-1, 1, 0, 0]
    # W: contributes to C11 (+1), C21 (+1)
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A11+A12)B22
    # U: A11 + A12 -> rows [1, 1, 0, 0]
    # V: B22 -> rows [0, 0, 0, 1]
    # W: contributes to C11 (-1), C12 (+1)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21 - A11 -> rows [-1, 0, 1, 0]
    # V: B11 + B12 -> rows [1, 1, 0, 0]
    # W: contributes to C22 (+1)
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12 - A22 -> rows [0, 1, 0, -1]
    # V: B21 + B22 -> rows [0, 0, 1, 1]
    # W: contributes to C11 (+1)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Block decomposition:
    # Split inner dimension m=3 into 1+2
    # Term 1: A[:,0:1] @ B[0:1,:] -> 2x1x4, rank 2
    # Term 2: A[:,1:3] @ B[1:3,:] -> 2x2x4
    #   Split p=4 into 2+2
    #   Term 2a: A[:,1:3] @ B[1:3,0:2] -> 2x2x2, rank 7 (Strassen)
    #   Term 2b: A[:,1:3] @ B[1:3,2:4] -> 2x2x2, rank 7 (Strassen)
    # Total rank: 2 + 7 + 7 = 16
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*4 = 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*2 = 8 rows
    
    col = 0
    
    # Term 1: A[:,0:1] @ B[0:1,:]
    # A indices: A[0,0] (U row 0), A[1,0] (U row 3)
    # B indices: B[0,0] (V row 0), B[0,1] (V row 1), B[0,2] (V row 2), B[0,3] (V row 3)
    # C indices: C[0,0] (W row 0), C[0,1] (W row 1), C[1,0] (W row 2), C[1,1] (W row 3)
    #            C[0,2] (W row 4), C[0,3] (W row 5), C[1,2] (W row 6), C[1,3] (W row 7)
    # 
    # C[0,0] += A[0,0]*B[0,0]
    # C[0,1] += A[0,0]*B[0,1]
    # C[0,2] += A[0,0]*B[0,2]
    # C[0,3] += A[0,0]*B[0,3]
    # C[1,0] += A[1,0]*B[0,0]
    # C[1,1] += A[1,0]*B[0,1]
    # C[1,2] += A[1,0]*B[0,2]
    # C[1,3] += A[1,0]*B[0,3]
    
    # This is 2x1x4, can be done with 2 rank-1 terms:
    # Term 1a: A[0,0] * B[0,:] -> contributes to C[0,:]
    #   U: [1,0,0,0,0,0], V: [1,1,1,1,0,0,0,0,0,0,0,0], W: [1,1,1,1,0,0,0,0]
    U[0][col] = 1  # A[0,0]
    V[0][col] = 1  # B[0,0]
    V[1][col] = 1  # B[0,1]
    V[2][col] = 1  # B[0,2]
    V[3][col] = 1  # B[0,3]
    W[0][col] = 1  # C[0,0]
    W[1][col] = 1  # C[0,1]
    W[4][col] = 1  # C[0,2]
    W[5][col] = 1  # C[0,3]
    col += 1
    
    # Term 1b: A[1,0] * B[0,:] -> contributes to C[1,:]
    #   U: [0,0,0,1,0,0], V: [1,1,1,1,0,0,0,0,0,0,0,0], W: [0,0,0,0,0,0,1,1]
    U[3][col] = 1  # A[1,0]
    V[0][col] = 1  # B[0,0]
    V[1][col] = 1  # B[0,1]
    V[2][col] = 1  # B[0,2]
    V[3][col] = 1  # B[0,3]
    W[2][col] = 1  # C[1,0]
    W[3][col] = 1  # C[1,1]
    W[6][col] = 1  # C[1,2]
    W[7][col] = 1  # C[1,3]
    col += 1
    
    # Term 2a: A[:,1:3] @ B[1:3,0:2] -> 2x2x2
    # A submatrix: A[0,1], A[0,2], A[1,1], A[1,2]
    #   U rows: 1 (A[0,1]), 2 (A[0,2]), 4 (A[1,1]), 5 (A[1,2])
    # B submatrix: B[1,0], B[1,1], B[2,0], B[2,1]
    #   V rows: 4 (B[1,0]), 5 (B[1,1]), 8 (B[2,0]), 9 (B[2,1])
    # C submatrix: C[0,0], C[0,1], C[1,0], C[1,1]
    #   W rows: 0 (C[0,0]), 1 (C[0,1]), 2 (C[1,0]), 3 (C[1,1])
    
    # Strassen terms for this 2x2x2 block
    strassen_terms_2a = _strassen_2x2x2(0, 0, 0, 0, 0, 0)
    u_row_map_2a = [1, 2, 4, 5]  # A11->1, A12->2, A21->4, A22->5
    v_row_map_2a = [4, 5, 8, 9]  # B11->4, B12->5, B21->8, B22->9
    w_row_map_2a = [0, 1, 2, 3]  # C11->0, C12->1, C21->2, C22->3
    
    for u_vec, v_vec, w_vec in strassen_terms_2a:
        for local_idx, u_val in enumerate(u_vec):
            if u_val != 0:
                U[u_row_map_2a[local_idx]][col] = u_val
        for local_idx, v_val in enumerate(v_vec):
            if v_val != 0:
                V[v_row_map_2a[local_idx]][col] = v_val
        for local_idx, w_val in enumerate(w_vec):
            if w_val != 0:
                W[w_row_map_2a[local_idx]][col] = w_val
        col += 1
    
    # Term 2b: A[:,1:3] @ B[1:3,2:4] -> 2x2x2
    # A submatrix: same as 2a
    #   U rows: 1 (A[0,1]), 2 (A[0,2]), 4 (A[1,1]), 5 (A[1,2])
    # B submatrix: B[1,2], B[1,3], B[2,2], B[2,3]
    #   V rows: 6 (B[1,2]), 7 (B[1,3]), 10 (B[2,2]), 11 (B[2,3])
    # C submatrix: C[0,2], C[0,3], C[1,2], C[1,3]
    #   W rows: 4 (C[0,2]), 5 (C[0,3]), 6 (C[1,2]), 7 (C[1,3])
    
    strassen_terms_2b = _strassen_2x2x2(0, 0, 0, 0, 0, 0)
    u_row_map_2b = [1, 2, 4, 5]  # Same A
    v_row_map_2b = [6, 7, 10, 11]  # B11->6, B12->7, B21->10, B22->11
    w_row_map_2b = [4, 5, 6, 7]  # C11->4, C12->5, C21->6, C22->7
    
    for u_vec, v_vec, w_vec in strassen_terms_2b:
        for local_idx, u_val in enumerate(u_vec):
            if u_val != 0:
                U[u_row_map_2b[local_idx]][col] = u_val
        for local_idx, v_val in enumerate(v_vec):
            if v_val != 0:
                V[v_row_map_2b[local_idx]][col] = v_val
        for local_idx, w_val in enumerate(w_vec):
            if w_val != 0:
                W[w_row_map_2b[local_idx]][col] = w_val
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for random A (2x3) and B (3x4), does sum_r U[:,r] * V[:,r] * W[:,r] == A @ B?
    # The tensor T[i,j,k] = delta[i*m+j, k*n+l] where C[l,i] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matrix multiplication C = A @ B:
    # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product term
    # 
    # U[i,j,r] * V[j,k,r] * W[k,n,i,r] summed over r should give delta terms
    # More precisely: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 
    # if (i,j,k) is a valid triple contributing to C[i,k]
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # u_col is indexed by i*m+j, v_col by j*p+k, w_col by k*n+i
        # Sum over i,j,k: u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        # This should equal C[i,k] for each i,k
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                C_reconstructed[i, k] += s
    
    max_err = np.max(np.abs(C_reconstructed - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
