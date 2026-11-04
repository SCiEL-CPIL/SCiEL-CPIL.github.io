"""Rewritten program with rank-81 decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Construct a rank-81 exact tensor decomposition for 4x4 by 4x6 matrix multiplication.
    
    Rank 81 = Rank(4x4x4) + Rank(4x4x2)
    Rank(4x4x4) = 49 (Recursive Strassen)
    Rank(4x4x2) = 32 (2 * Rank(4x4x1) = 2 * 16)
    """
    start = time.time()
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    rank = 81
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Part 1: 4x4 by 4x4 (Rank 49) ---
    # Indices for U: 0..15
    # Indices for V: 0..15 (cols 0-3 of B)
    # Indices for W: 0..15 (rows 0-15, corresponding to k=0..3)
    
    # Define Strassen 2x2x2 steps
    # U, V, W are 2x2 matrices of coefficients
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11(B10-B00)
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A (2x2), B (2x2), C (2x2)
    # Note: The C part accumulates.
    # We store U_k, V_k for the k-th product.
    # And we need to know how C is formed from M_1..M_7.
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # Let's represent the 7 steps as (U_2x2, V_2x2)
    # and then map them to C_2x2.
    
    S_U = [
        [[1, 0], [0, 1]],  # M1: A00 + A11
        [[0, 0], [1, 1]],  # M2: A10 + A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]]  # M7: A01 - A11
    ]
    S_V = [
        [[1, 0], [0, 1]],  # M1: B00 + B11
        [[1, 0], [0, 0]],  # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[0, -1], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]],  # M5: B11
        [[1, 1], [0, 0]],  # M6: B00 + B01
        [[0, 1], [1, 1]]   # M7: B10 + B11
    ]
    
    # C coefficients: C[i][j] = sum_k c_k[i][j] * M_k
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1
    # M4: C00 -= 1, C10 += 1
    # M5: C00 += 1
    # M6: C11 += 1
    # M7: C00 -= 1, C01 += 1
    
    S_C = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 0]],  # M3
        [[-1, 0], [1, 0]], # M4
        [[1, 0], [0, 0]],  # M5
        [[0, 0], [0, 1]],  # M6
        [[-1, 1], [0, 0]]  # M7
    ]
    
    col_idx = 0
    for k in range(7): # Outer Strassen step
        # U_outer[k] is a 2x2 matrix of coefficients for the 4 blocks of A
        # V_outer[k] is a 2x2 matrix of coefficients for the 4 blocks of B
        U_outer = S_U[k]
        V_outer = S_V[k]
        
        for s in range(7): # Inner Strassen step
            U_inner = S_U[s]
            V_inner = S_V[s]
            C_inner = S_C[s]
            
            # Compose U
            # U_global[row, col] += coeff
            # A is 4x4. Blocks A00, A01, A10, A11.
            # U_outer[i][j] is coeff for block A_ij in the outer step.
            # U_inner[r][c] is coeff for entry (r,c) of the 2x2 matrix X.
            # X is formed by summing blocks.
            # Wait, X is a 2x2 matrix. X_rc = sum_{i,j} U_outer[i][j] * A_ij?
            # No.
            # The Strassen formula for 4x4x4:
            # M1 = (A00 + A11)(B00 + B11).
            # Here X = A00 + A11. X is a 2x2 matrix.
            # X_rc = (A00)_rc + (A11)_rc.
            # So X_rc is a linear combination of entries of A00 and A11.
            # Specifically, X_rc = 1*A00_rc + 0*A01_rc + 0*A10_rc + 1*A11_rc.
            # So the coefficient for A_ij_rc in X_rc is:
            # coeff_A[i][j] for the block, and we pick entry (r,c) of that block.
            
            # In our U_global, row index is i*m + j = i*4 + j.
            # Block A_ij has rows 2i..2i+1 and cols 2j..2j+1.
            # Entry (r,c) of block A_ij is at global row (2i+r)*4 + (2j+c).
            
            # The inner step s computes x_s = sum_{r,c} U_inner[r][c] * X_rc.
            # So x_s = sum_{r,c} U_inner[r][c] * (sum_{i,j} U_outer[i][j] * A_ij_rc ?)
            # No, U_outer[i][j] is the coefficient for the block A_ij in the sum forming X.
            # X = sum_{i,j} U_outer[i][j] A_ij.
            # So X_rc = sum_{i,j} U_outer[i][j] A_ij_rc.
            # Thus x_s = sum_{r,c} U_inner[r][c] sum_{i,j} U_outer[i][j] A_ij_rc
            #          = sum_{i,j,r,c} (U_outer[i][j] * U_inner[r][c]) A_ij_rc.
            
            # So for each i, j, r, c:
            for i in range(2):
                for j in range(2):
                    coeff_outer = U_outer[i][j]
                    if coeff_outer == 0: continue
                    for r in range(2):
                        for c in range(2):
                            coeff_inner = U_inner[r][c]
                            if coeff_inner == 0: continue
                            total_coeff = coeff_outer * coeff_inner
                            if total_coeff == 0: continue
                            
                            global_row = (2*i + r) * 4 + (2*j + c)
                            U[global_row, col_idx] += total_coeff
            
            # Compose V
            # Y = sum_{i,j} V_outer[i][j] B_ij.
            # y_s = sum_{r,c} V_inner[r][c] Y_rc.
            # Same logic.
            for i in range(2):
                for j in range(2):
                    coeff_outer = V_outer[i][j]
                    if coeff_outer == 0: continue
                    for r in range(2):
                        for c in range(2):
                            coeff_inner = V_inner[r][c]
                            if coeff_inner == 0: continue
                            total_coeff = coeff_outer * coeff_inner
                            if total_coeff == 0: continue
                            
                            # V row: j*m + k = j*4 + k.
                            # Block B_ij has rows 2i..2i+1? No.
                            # B is 4x4.
                            # B_ij is the block at row i, col j of the 2x2 block structure.
                            # Row index in B: 2i+r. Col index in B: 2j+c.
                            # Global V row: (2i+r)*4 + (2j+c).
                            global_row = (2*i + r) * 4 + (2*j + c)
                            V[global_row, col_idx] += total_coeff
            
            # Compose W
            # C_block = sum_k S_C[k] M_k.
            # Here we are at outer step k, inner step s.
            # The result of the inner step s is a scalar contribution to the entries of the 2x2 matrix result of M_k.
            # Let R be the 2x2 matrix result of M_k.
            # R_rc = sum_s C_inner_s[r][c] * (M_k)_rc?
            # No.
            # The inner Strassen decomposition computes the 2x2 matrix product X Y = Z.
            # Z_rc = sum_s C_inner[s][r][c] * M_s?
            # Wait. In the inner Strassen, we compute 7 scalars M_s.
            # Then Z_00 = M_0 - M_3 + M_4 - M_6.
            # So Z_rc = sum_s C_coeff[s][r][c] M_s.
            # Here M_s is the scalar product x_s y_s.
            # So Z_rc is a linear combination of the 7 inner products.
            # But in the rank-1 decomposition, each inner step s provides one rank-1 term.
            # The term is x_s * y_s * C_coeff[s][r][c] * (Block in C).
            # Which block in C?
            # The outer step k produces a matrix M_k (which is Z in this context).
            # This matrix M_k contributes to the blocks of C.
            # C_block_ij += C_outer[k][i][j] * M_k.
            # So for the global C matrix, the contribution from this term (k,s) is:
            # For each block (i,j) of C, the coefficient is C_outer[k][i][j] * C_inner[s][r][c] ?
            # No.
            # Z_rc is the (r,c) entry of the 2x2 matrix M_k.
            # M_k is added to C_block_ij with coefficient C_outer[k][i][j].
            # So C_block_ij_rc += C_outer[k][i][j] * Z_rc.
            # And Z_rc = sum_s C_inner[s][r][c] (x_s y_s).
            # So for term (k,s), the coefficient for C_block_ij_rc is C_outer[k][i][j] * C_inner[s][r][c].
            
            # We need C_outer (S_C for the outer level).
            # We already defined S_C for the 2x2x2 Strassen.
            # So C_outer[k] = S_C[k].
            
            C_outer = S_C[k]
            
            for i in range(2):
                for j in range(2):
                    coeff_outer = C_outer[i][j]
                    if coeff_outer == 0: continue
                    for r in range(2):
                        for c in range(2):
                            coeff_inner = C_inner[r][c]
                            if coeff_inner == 0: continue
                            total_coeff = coeff_outer * coeff_inner
                            if total_coeff == 0: continue
                            
                            # W row: k*n + i = k*4 + i.
                            # Block C_ij has rows 2i..2i+1, cols 2j..2j+1.
                            # Entry (r,c) of C_ij is at global row (2j+c)*4 + (2i+r).
                            # Wait. W index is k*n + i.
                            # C is 4x4. C[row, col].
                            # Block C_ij is at row 2i..2i+1, col 2j..2j+1.
                            # Entry is C[2i+r, 2j+c].
                            # W row index = col * n + row = (2j+c)*4 + (2i+r).
                            global_row = (2*j + c) * 4 + (2*i + r)
                            W[global_row, col_idx] += total_coeff
            
            col_idx += 1
            
    # --- Part 2: 4x4 by 4x2 (Rank 32) ---
    # Columns 4 and 5 of B.
    # For each column k in {4, 5}:
    # 16 terms (i, j) where i in 0..3, j in 0..3.
    # U[i*4 + j, col] += 1
    # V[j*6 + k, col] += 1
    # W[k*4 + i, col] += 1
    
    for k in [4, 5]:
        for i in range(4):
            for j in range(4):
                u_row = i * 4 + j
                v_row = j * 6 + k
                w_row = k * 4 + i
                U[u_row, col_idx] = 1.0
                V[v_row, col_idx] = 1.0
                W[w_row, col_idx] = 1.0
                col_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify
    n, m, p = 4, 4, 6
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    
    # Tensor reconstruction
    # C[i, k] = sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r]
    # This sum is over j as well?
    # The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # C[i,k] = sum_j sum_r U_{i,m,j,r} V_{m,p,j,k,r} W_{p,n,k,i,r}
    
    C_calc = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # u is 16x1, v is 24x1, w is 24x1
        # Reshape u to 4x4
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        
        # Contribution: sum_j u_mat[i,j] v_mat[j,k] w_mat[k,i]
        # This is trace(u_mat * v_mat * w_mat * I)? No.
        # It is element-wise sum.
        # C_contrib[i,k] = sum_j u_mat[i,j] v_mat[j,k] w_mat[k,i]
        
        # Can be computed as:
        # temp = u_mat * v_mat (4x6)
        # C_contrib = temp * w_mat.T (w_mat is 6x4, so w_mat.T is 4x6)
        # w_mat[k,i] is element (k,i) of w_mat.
        # So w_mat.T[i,k] = w_mat[k,i].
        
        temp = u_mat * v_mat # 4x6
        c_contrib = temp * w_mat.T # 4x6
        C_calc += c_contrib
        
    C_true = A @ B
    diff = np.max(np.abs(C_calc - C_true))
    print(f"Max diff: {diff}")
    print(f"Rank: {U.shape[1]}")
    print(f"Score: {96 / U.shape[1]}")
