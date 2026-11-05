import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_222():
        # Rank 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: i*2+j where i in {0,1}, j in {0,1}
        # B indices: j*2+k where j in {0,1}, k in {0,1}
        # C indices: k*2+i where k in {0,1}, i in {0,1}
        
        # M1: (A00+A11)(B00+B11)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
        
        # M2: (A10+A11)B00
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 0
        
        # M3: A00(B10-B11)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 1; W[3, 2] = 0
        
        # M4: A11(B01-B00)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [-1, 1, 0, 0]
        W[0, 3] = 1; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
        
        # M5: (A00+A01)B11
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[0, 4] = -1; W[1, 4] = 0; W[2, 4] = 0; W[3, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
        
        return U, V, W, 7
    
    # Helper to get schoolbook 2x2x1 decomposition
    def get_schoolbook_221():
        # 2x2x1: rank 4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                U[i * 2 + j, col] = 1.0
                V[j, col] = 1.0
                W[i, col] = 1.0
                col += 1
        return U, V, W, 4
    
    # Build the decomposition using block decomposition
    # 4x5x6 = (4x4x4) + (4x4x2) + (4x1x6)
    # Using Strassen for 2x2x2 blocks
    
    U_s, V_s, W_s, R_s = get_strassen_222()
    U_sb, V_sb, W_sb, R_sb = get_schoolbook_221()
    
    # Component 1: 4x4x4 = Strassen ⊗ Strassen, rank 49
    # A is 4x4, B is 4x4, C is 4x4
    # A[i,j] with i = i1*2+i0, j = j1*2+j0
    # B[j,k] with j = j1*2+j0, k = k1*2+k0
    # C[k,i] with k = k1*2+k0, i = i1*2+i0
    # U_c1 = kron(U_s, U_s): row (i1*2+i0, j1*2+j0) -> index (i1*2+j1)*2 + (i0*2+j0)
    # Actually np.kron(U_s, U_s) has row index = i_outer * 4 + i_inner
    # where i_outer is from first U_s (i1*2+j1) and i_inner is from second U_s (i0*2+j0)
    # So row = (i1*2+j1)*4 + (i0*2+j0)
    # Global U row for A[i,j] where i=i1*2+i0, j=j1*2+j0: i*5+j = (i1*2+i0)*5 + j1*2+j0
    
    U_c1 = np.kron(U_s, U_s)  # shape (16, 49)
    V_c1 = np.kron(V_s, V_s)  # shape (16, 49)
    W_c1 = np.kron(W_s, W_s)  # shape (16, 49)
    R_c1 = 49
    
    # Component 2: 4x4x2 = Strassen ⊗ Schoolbook(2x2x1), rank 28
    # A is 4x4, B is 4x2, C is 4x2
    # A[i,j] with i = i1*2+i0, j = j1*2+j0
    # B[j,k] with j = j1*2+j0, k = k0 (0 or 1)
    # C[k,i] with k = k0, i = i1*2+i0
    # U_c2 = kron(U_s, U_sb): shape (16, 28)
    # V_c2 = kron(V_s, V_sb): shape (16, 28) -- wait, V_s is (4,7), V_sb is (2,4)
    # kron(V_s, V_sb) has shape (4*2, 7*4) = (8, 28)
    # Hmm, that's only 8 rows. Let me reconsider.
    # 
    # For the 4x4x2 tensor:
    # The "4" in rows of A comes from 2x2 blocks (Strassen structure)
    # The "4" in cols of A comes from 2x2 blocks
    # The "2" in cols of B comes from the single k dimension
    # 
    # Actually, let me think of it as: we have a 4x4x2 tensor.
    # We can view it as 2x2 blocks in i and j (Strassen), and 2 in k.
    # So it's (2x2) x (2x2) x 2 = Strassen(2x2x2) with the 2nd dimension of B being size 2.
    # 
    # Alternative: view as (2x2) in i, (2x2) in j, (1x2) in k
    # = Schoolbook(2x2x1) in (i_block, j_block, k_block) ⊗ Strassen(2x2x2) in (i0, j0, k0)
    # No wait, that doesn't work cleanly either.
    #
    # Let me just use a direct approach: 4x4x2 can be decomposed as
    # 2x2x2 (Strassen) applied to each of the 2 k-values, but that's not right either.
    #
    # Actually, the cleanest way: 4x4x2 = (2x2x2) + (2x2x2) where we split k into two halves?
    # No, k only has 2 values.
    #
    # Let me think differently. The 4x4x2 tensor T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..1.
    # We can decompose this as: for each k in {0,1}, we have a 4x4x1 tensor.
    # 4x4x1 can be done with rank 16 (schoolbook) or better.
    # Actually 4x4x1: T[i,j,0] = 1. This is just the all-ones 4x4 matrix.
    # Rank of all-ones 4x4 matrix is 1! So 4x4x1 has rank 1.
    # Wait no, the tensor T[i,j,k] for fixed k is a 4x4 matrix of all 1s, which has rank 1.
    # So 4x4x2 = 4x4x1 (k=0) + 4x4x1 (k=1), each rank 1, total rank 2?
    # No, that's not how tensor rank works. The tensor rank is not the sum of slice ranks.
    #
    # Let me just use the Kronecker product approach properly.
    # 4x4x2 can be seen as: i = i1*2+i0, j = j1*2+j0, k = k0
    # T[i,j,k] = T[i1,i0,j1,j0,k0]
    # If we use Strassen in (i0,j0,k0) and schoolbook in (i1,j1,k0)... no.
    #
    # Simplest correct approach: use Strassen for the 2x2x2 sub-problems.
    # Split into 4 blocks of 2x2x2: (i1,j1) in {0,1}^2, each with k in {0,1}
    # Wait, that gives 4 blocks of 2x2x2, each rank 7, total 28. That works!
    # But we need to handle the indexing correctly.
    #
    # Actually even simpler: just use the Kronecker product of Strassen(2x2x2) and a 2x1x1 identity-like thing.
    # Hmm, this is getting complicated. Let me just directly construct 4x4x2.
    
    # For 4x4x2: use Strassen for each pair of (i0,j0) blocks
    # i = i1*2 + i0, j = j1*2 + j0, k = k0
    # For fixed (i1,j1), we have a 2x2x2 sub-tensor in (i0,j0,k0)
    # There are 4 such sub-tensors, each rank 7, total rank 28
    
    # U_c2: shape (16, 28), row = i*4+j = (i1*2+i0)*4 + (j1*2+j0)
    # V_c2: shape (8, 28), row = j*2+k = (j1*2+j0)*2 + k0
    # W_c2: shape (8, 28), row = k*4+i = k0*4 + (i1*2+i0)
    
    U_c2 = np.zeros((16, 28))
    V_c2 = np.zeros((8, 28))
    W_c2 = np.zeros((8, 28))
    R_c2 = 28
    
    for i1 in range(2):
        for j1 in range(2):
            base_r = (i1 * 2 + j1) * 7
            for r_s in range(7):
                r = base_r + r_s
                # U: for i0 in 0,1 and j0 in 0,1
                for i0 in range(2):
                    for j0 in range(2):
                        local_a_idx = i0 * 2 + j0
                        global_i = i1 * 2 + i0
                        global_j = j1 * 2 + j0
                        global_u_row = global_i * 4 + global_j
                        U_c2[global_u_row, r] = U_s[local_a_idx, r_s]
                # V: for j0 in 0,1 and k0 in 0,1
                for j0 in range(2):
                    for k0 in range(2):
                        local_b_idx = j0 * 2 + k0
                        global_j = j1 * 2 + j0
                        global_v_row = global_j * 2 + k0
                        V_c2[global_v_row, r] = V_s[local_b_idx, r_s]
                # W: for k0 in 0,1 and i0 in 0,1
                for k0 in range(2):
                    for i0 in range(2):
                        local_c_idx = k0 * 2 + i0
                        global_i = i1 * 2 + i0
                        global_w_row = k0 * 4 + global_i
                        W_c2[global_w_row, r] = W_s[local_c_idx, r_s]
    
    # Component 3: 4x1x6, rank 24 (schoolbook)
    U_c3, V_c3, W_c3, R_c3 = get_schoolbook(4, 1, 6)
    
    Total_Rank = R_c1 + R_c2 + R_c3  # 49 + 28 + 24 = 101
    
    U_global = np.zeros((n * m, Total_Rank))
    V_global = np.zeros((m * p, Total_Rank))
    W_global = np.zeros((p * n, Total_Rank))
    
    # Fill Component 1: 4x4x4
    # Global U row for A[i,j]: i*5+j, where i in 0..3, j in 0..3
    # U_c1 row: (i1*2+j1)*4 + (i0*2+j0) where i=i1*2+i0, j=j1*2+j0
    # Global U row: i*5+j = (i1*2+i0)*5 + j1*2+j0
    
    for r in range(R_c1):
        for i1 in range(2):
            for i0 in range(2):
                for j1 in range(2):
                    for j0 in range(2):
                        local_u_row = (i1 * 2 + j1) * 4 + (i0 * 2 + j0)
                        global_i = i1 * 2 + i0
                        global_j = j1 * 2 + j0
                        global_u_row = global_i * 5 + global_j
                        U_global[global_u_row, r] = U_c1[local_u_row, r]
        
        # V_c1 row: (j1*2+k1)*4 + (j0*2+k0) where j=j1*2+j0, k=k1*2+k0
        # Global V row for B[j,k]: j*6+k = (j1*2+j0)*6 + k1*2+k0
        for j1 in range(2):
            for j0 in range(2):
                for k1 in range(2):
                    for k0 in range(2):
                        local_v_row = (j1 * 2 + k1) * 4 + (j0 * 2 + k0)
                        global_j = j1 * 2 + j0
                        global_k = k1 * 2 + k0
                        global_v_row = global_j * 6 + global_k
                        V_global[global_v_row, r] = V_c1[local_v_row, r]
        
        # W_c1 row: (k1*2+i1)*4 + (k0*2+i0) where k=k1*2+k0, i=i1*2+i0
        # Global W row for C[k,i]: k*4+i = (k1*2+k0)*4 + i1*2+i0
        for k1 in range(2):
            for k0 in range(2):
                for i1 in range(2):
                    for i0 in range(2):
                        local_w_row = (k1 * 2 + i1) * 4 + (k0 * 2 + i0)
                        global_k = k1 * 2 + k0
                        global_i = i1 * 2 + i0
                        global_w_row = global_k * 4 + global_i
                        W_global[global_w_row, r] = W_c1[local_w_row, r]
    
    # Fill Component 2: 4x4x2
    # A[i,j] with i in 0..3, j in 0..3 -> Global U row: i*5+j
    # B[j,k] with j in 0..3, k in 4..5 -> Global V row: j*6+k = j*6+(4+k0)
    # C[k,i] with k in 4..5, i in 0..3 -> Global W row: k*4+i = (4+k0)*4+i
    
    offset_r = R_c1
    for r in range(R_c2):
        # U_c2 row: i*4+j where i in 0..3, j in 0..3
        for i in range(4):
            for j in range(4):
                local_u_row = i * 4 + j
                global_u_row = i * 5 + j
                U_global[global_u_row, offset_r + r] = U_c2[local_u_row, r]
        
        # V_c2 row: j*2+k0 where j in 0..3, k0 in 0..1
        for j in range(4):
            for k0 in range(2):
                local_v_row = j * 2 + k0
                global_k = 4 + k0
                global_v_row = j * 6 + global_k
                V_global[global_v_row, offset_r + r] = V_c2[local_v_row, r]
        
        # W_c2 row: k0*4+i where k0 in 0..1, i in 0..3
        for k0 in range(2):
            for i in range(4):
                local_w_row = k0 * 4 + i
                global_k = 4 + k0
                global_w_row = global_k * 4 + i
                W_global[global_w_row, offset_r + r] = W_c2[local_w_row, r]
    
    # Fill Component 3: 4x1x6
    # A[i,4] for i in 0..3 -> Global U row: i*5+4
    # B[4,k] for k in 0..5 -> Global V row: 4*6+k = 24+k
    # C[k,i] for k in 0..5, i in 0..3 -> Global W row: k*4+i
    
    offset_r = R_c1 + R_c2
    for i in range(4):
        U_global[i * 5 + 4, offset_r:offset_r + R_c3] = U_c3[i, :]
    for k in range(6):
        V_global[4 * 6 + k, offset_r:offset_r + R_c3] = V_c3[k, :]
    W_global[0:24, offset_r:offset_r + R_c3] = W_c3[:, :]
    
    # Verify
    U_r = U_global.reshape(n, m, Total_Rank)
    V_r = V_global.reshape(m, p, Total_Rank)
    W_r = W_global.reshape(p, n, Total_Rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    if np.allclose(T_recon, T_target, atol=1e-9):
        return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    else:
        # Fallback to schoolbook
        U_school, V_school, W_school, R_school = get_schoolbook(n, m, p)
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
