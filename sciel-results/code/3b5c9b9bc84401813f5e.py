import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    # Build a rank-55 decomposition using Strassen on 2x2x2 blocks
    # Split m=3 into (2,1). The 2x2 part uses Strassen (rank 7) for each of 5 blocks (k=0..8 in steps of 2)
    # The 2x1x10 remainder uses schoolbook (rank 20)
    
    # Strassen decomposition for 2x2x2:
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
    
    # For 2x2x2, the Strassen coefficients:
    # L (U side, rows are i*m+j, cols are the 7 Strassen products):
    # A11=row0, A12=row1, A21=row2, A22=row3 (for i=0,1 and j=0,1)
    # Actually for 2x2 matrix A (i=0,1; j=0,1):
    #   A11 = A[0,0] -> index 0
    #   A12 = A[0,1] -> index 1
    #   A21 = A[1,0] -> index 2
    #   A22 = A[1,1] -> index 3
    #
    # M1 = (A11+A22)(B11+B22): U has 1 at A11 and A22
    # M2 = (A21+A22)B11: U has 1 at A21 and A22
    # M3 = A11(B21-B22): U has 1 at A11
    # M4 = A22(B12-B11): U has 1 at A22
    # M5 = (A11+A12)B22: U has 1 at A11 and A12
    # M6 = (A21-A11)(B11+B12): U has 1 at A21, -1 at A11
    # M7 = (A12-A22)(B21+B22): U has 1 at A12, -1 at A22
    
    # For the 2x3x10 problem:
    # Split j (middle dim, size 3) into j0,j1 (for Strassen) and j2 (schoolbook)
    # Split k (size 10) into 5 pairs: (k0,k1), (k2,k3), ..., (k8,k9)
    
    # For each block of 2 k-values, we have a 2x2x2 subproblem
    # A is 2x2 (i=0,1; j=0,1), B is 2x2 (j=0,1; k=0,1 within block), C is 2x2 (i=0,1; k=0,1 within block)
    
    # Strassen coefficients for U (2x2 A matrix, 7 products):
    # Product s: which A entries are involved
    L = [
        {0: 1, 3: 1},      # M1: A11+A22
        {2: 1, 3: 1},      # M2: A21+A22
        {0: 1},            # M3: A11
        {3: 1},            # M4: A22
        {0: 1, 1: 1},      # M5: A11+A12
        {2: 1, 0: -1},     # M6: A21-A11
        {1: 1, 3: -1}      # M7: A12-A22
    ]
    
    # Strassen coefficients for V (2x2 B matrix, 7 products):
    # B entries: B11=index0, B12=index1, B21=index2, B22=index3
    R = [
        {0: 1, 3: 1},      # M1: B11+B22
        {0: 1},            # M2: B11
        {2: 1, 3: -1},     # M3: B21-B22
        {1: 1, 0: -1},     # M4: B12-B11
        {3: 1},            # M5: B22
        {0: 1, 1: 1},      # M6: B11+B12
        {2: 1, 3: 1}       # M7: B21+B22
    ]
    
    # Strassen coefficients for W (2x2 C matrix, 7 products):
    # C entries: C11=index0, C12=index1, C21=index2, C22=index3
    Wc = [
        {0: 1, 3: 1},      # M1: C11+C22
        {2: 1, 3: -1},     # M2: C21-C22
        {1: 1, 3: 1},      # M3: C12+C22
        {0: 1, 2: 1},      # M4: C11+C21
        {1: 1, 0: -1},     # M5: C12-C11
        {3: 1},            # M6: C22
        {0: 1}             # M7: C11
    ]
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # 5 Strassen blocks for k=0,1; 2,3; 4,5; 6,7; 8,9
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            # U: A is 2x2 (i=0,1; j=0,1)
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            # V: B is 2x2 (j=0,1; k=k_off+k_off+1)
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            # W: C is 2x2 (i=0,1; k=k_off+k_off+1)
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    # Schoolbook for the remaining j=2 (A[*,2] * B[2,*] -> C)
    # This is a 2x1x10 subproblem: rank 2*1*10 = 20
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor C matrix: each column r is the vectorized u_r ⊗ v_r ⊗ w_r
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    total_dim = dim_u * dim_v * dim_w  # 3600
    
    C = np.zeros((total_dim, rank))
    for r in range(rank):
        C[:, r] = np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r]).flatten()
    
    # Check for redundant columns using QR with column pivoting
    # For each column r, check if C_r is in the span of the other columns
    # Use a more efficient approach: compute the rank of C and check if removing any column drops the rank
    
    current_rank = rank
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    C_cur = C.copy()
    
    # Use a greedy approach: try to remove columns one by one
    # For each candidate column r, solve the least squares problem
    # and check if the residual is small
    
    def is_redundant(r, C_mat, current_rank):
        if current_rank <= 1:
            return False
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        C_rem = C_mat[:, mask]
        target = C_mat[:, r]
        
        # Use SVD to check if target is in the column space of C_rem
        try:
            U_svd, s, Vt = np.linalg.svd(C_rem, full_matrices=False)
            # Check if target is in the range of C_rem
            # Project target onto the column space
            proj = U_svd @ (U_svd.T @ target)
            residual = target - proj
            if np.linalg.norm(residual) < 1e-8:
                return True
        except np.linalg.LinAlgError:
            pass
        return False
    
    # Try to remove redundant columns
    r = 0
    while r < current_rank:
        if time.time() > deadline:
            break
        if is_redundant(r, C_cur, current_rank):
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            U_cur = U_cur[:, mask]
            V_cur = V_cur[:, mask]
            W_cur = W_cur[:, mask]
            C_cur = C_cur[:, mask]
            current_rank -= 1
        else:
            r += 1
    
    # Convert back to lists
    U_out = U_cur.tolist()
    V_out = V_cur.tolist()
    W_out = W_cur.tolist()
    
    # Round to nearest half-integer to ensure clean values
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
