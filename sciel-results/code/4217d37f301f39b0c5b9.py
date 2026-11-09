import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = 3, 4, 7
    R = 78
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Helper to add a scalar multiplication
    # A[i, j] * B[j, k] -> C[k, i]
    def add_mult(i, j, k, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal col
        if col >= R: return
        u_idx = i*m + j
        v_idx = j*p + k
        w_idx = k*n + i
        U[u_idx, col] += coeff_u
        V[v_idx, col] += coeff_v
        W[w_idx, col] += coeff_w
        col += 1

    # Helper to add Strassen block
    # A_block at global (ai, aj), B_block at global (bj, bk), C_block at global (ci, ck)
    # Local 2x2x2
    # ai, aj: top-left of A block (2x2)
    # bj, bk: top-left of B block (2x2)
    # ci, ck: top-left of C block (2x2)
    def add_strassen(ai, aj, bj, bk, ci, ck):
        nonlocal col
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        # U: A[ai,aj], A[ai+1,aj+1]
        # V: B[bj,bk], B[bj+1,bk+1]
        # W: C[ck,ci] (+1), C[ck+1, ci+1] (+1)
        
        # We need to add 7 tensors.
        
        # 1
        # A: (ai, aj), (ai+1, aj+1)
        # B: (bj, bk), (bj+1, bk+1)
        # C: (ck, ci) coeff 1, (ck+1, ci+1) coeff 1
        U[ai*m + aj, col] += 1
        U[(ai+1)*m + (aj+1), col] += 1
        V[bj*p + bk, col] += 1
        V[(bj+1)*p + (bk+1), col] += 1
        W[ck*n + ci, col] += 1
        W[(ck+1)*n + (ci+1), col] += 1
        col += 1
        
        # 2
        # M2 = (A21+A22)B11 -> C21, C22 (-1)
        # A: (ai+1, aj), (ai+1, aj+1)
        # B: (bj, bk)
        # C: (ck, ci+1) coeff 1, (ck+1, ci+1) coeff -1
        U[(ai+1)*m + aj, col] += 1
        U[(ai+1)*m + (aj+1), col] += 1
        V[bj*p + bk, col] += 1
        W[ck*n + (ci+1), col] += 1
        W[(ck+1)*n + (ci+1), col] += -1
        col += 1
        
        # 3
        # M3 = A11(B21-B22) -> C12, C22
        # A: (ai, aj)
        # B: (bj+1, bk), (bj+1, bk+1)
        # C: (ck+1, ci) coeff 1, (ck+1, ci+1) coeff 1
        U[ai*m + aj, col] += 1
        V[(bj+1)*p + bk, col] += 1
        V[(bj+1)*p + (bk+1), col] += -1
        W[(ck+1)*n + ci, col] += 1
        W[(ck+1)*n + (ci+1), col] += 1
        col += 1
        
        # 4
        # M4 = A22(B12-B11) -> C11 (-1), C21
        # A: (ai+1, aj+1)
        # B: (bj, bk+1), (bj, bk)
        # C: (ck, ci) coeff -1, (ck, ci+1) coeff 1
        U[(ai+1)*m + (aj+1), col] += 1
        V[bj*p + (bk+1), col] += 1
        V[bj*p + bk, col] += -1
        W[ck*n + ci, col] += -1
        W[ck*n + (ci+1), col] += 1
        col += 1
        
        # 5
        # M5 = (A11+A12)B22 -> C11 (-1), C12
        # A: (ai, aj), (ai, aj+1)
        # B: (bj+1, bk+1)
        # C: (ck, ci) coeff -1, (ck+1, ci) coeff 1
        U[ai*m + aj, col] += 1
        U[ai*m + (aj+1), col] += 1
        V[(bj+1)*p + (bk+1), col] += 1
        W[ck*n + ci, col] += -1
        W[(ck+1)*n + ci, col] += 1
        col += 1
        
        # 6
        # M6 = (A21-A11)(B11+B12) -> C22
        # A: (ai+1, aj), (ai, aj)
        # B: (bj, bk), (bj, bk+1)
        # C: (ck+1, ci+1) coeff 1
        U[(ai+1)*m + aj, col] += 1
        U[ai*m + aj, col] += -1
        V[bj*p + bk, col] += 1
        V[bj*p + (bk+1), col] += 1
        W[(ck+1)*n + (ci+1), col] += 1
        col += 1
        
        # 7
        # M7 = (A12-A22)(B21+B22) -> C11
        # A: (ai, aj+1), (ai+1, aj+1)
        # B: (bj+1, bk), (bj+1, bk+1)
        # C: (ck, ci) coeff 1
        U[ai*m + (aj+1), col] += 1
        U[(ai+1)*m + (aj+1), col] += -1
        V[(bj+1)*p + bk, col] += 1
        V[(bj+1)*p + (bk+1), col] += 1
        W[ck*n + ci, col] += 1
        col += 1

    # Top row: C_1 = A_11 B_1 + A_12 B_2
    # A_11 is A[0, 0:2], B_1 is B[0:2, :7]
    # A_12 is A[0, 2:4], B_2 is B[2:4, :7]
    
    # A_11 B_1: 1x2 * 2x7
    for k in range(7):
        # j=0
        add_mult(0, 0, k)
        # j=1
        add_mult(0, 1, k)
        
    # A_12 B_2: 1x2 * 2x7
    for k in range(7):
        # j=2
        add_mult(0, 2, k)
        # j=3
        add_mult(0, 3, k)
        
    # Bottom part: C_2 = A_21 B_1 + A_22 B_2
    # A_21 is A[1:3, 0:2], B_1 is B[0:2, :7]
    # A_22 is A[1:3, 2:4], B_2 is B[2:4, :7]
    
    # Process A_21 B_1
    # Split B_1 into 3 blocks of 2 cols and 1 block of 1 col
    # Block 1: cols 0,1
    add_strassen(1, 0, 0, 0, 1, 0) # A[1:3, 0:2], B[0:2, 0:2], C[1:3, 0:2]
    # Block 2: cols 2,3
    add_strassen(1, 0, 0, 2, 1, 2)
    # Block 3: cols 4,5
    add_strassen(1, 0, 0, 4, 1, 4)
    # Block 4: col 6
    # Schoolbook 2x2x1
    # j=0,1
    # k=6
    add_mult(1, 0, 6)
    add_mult(2, 0, 6)
    add_mult(1, 1, 6)
    add_mult(2, 1, 6)
    
    # Process A_22 B_2
    # Block 1: cols 0,1
    add_strassen(1, 2, 2, 0, 1, 0) # A[1:3, 2:4], B[2:4, 0:2], C[1:3, 0:2]
    # Block 2: cols 2,3
    add_strassen(1, 2, 2, 2, 1, 2)
    # Block 3: cols 4,5
    add_strassen(1, 2, 2, 4, 1, 4)
    # Block 4: col 6
    add_mult(1, 2, 6)
    add_mult(2, 2, 6)
    add_mult(1, 3, 6)
    add_mult(2, 3, 6)

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
