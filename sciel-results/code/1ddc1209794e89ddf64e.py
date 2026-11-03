import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    rank_school = n * m * p  # 56

    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit

    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)

    def clean_half_int(x):
        x2 = x * 2
        return np.round(x2) / 2

    # Build the multiplication tensor T as a flat vector
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0

    # Schoolbook solution as fallback
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    def strassen_2x2x2(U_mat, V_mat, W_mat, base_col, ac0, br0, bc0, cc0, cr0):
        """
        Add Strassen terms for computing a 2x2 * 2x2 block.
        A contributes rows cr0 to cr0+1, cols ac0 to ac0+1
        B contributes rows br0 to br0+1, cols bc0 to bc0+1
        C gets rows cr0 to cr0+1, cols cc0 to cc0+1
        """
        # U indices: A[cr0+i, ac0+j] -> index (cr0+i)*m + (ac0+j)
        # V indices: B[br0+j, bc0+k] -> index (br0+j)*p + (bc0+k)
        # W indices: C[cr0+i, cc0+k] -> index (cc0+k)*n + (cr0+i)
        
        terms = []
        
        # P1 = (a00+a11)*(b00+b11)
        u1 = {(cr0)*m + ac0: 1, (cr0+1)*m + (ac0+1): 1}
        v1 = {br0*p + bc0: 1, (br0+1)*p + (bc0+1): 1}
        w1 = {(cc0)*n + cr0: 1, (cc0+1)*n + (cr0+1): 1}
        terms.append((u1, v1, w1))
        
        # P2 = (a10+a11)*b00
        u2 = {(cr0+1)*m + ac0: 1, (cr0+1)*m + (ac0+1): 1}
        v2 = {br0*p + bc0: 1}
        w2 = {(cc0)*n + (cr0+1): 1}
        terms.append((u2, v2, w2))
        
        # P3 = a00 * (b01-b11)
        u3 = {(cr0)*m + ac0: 1}
        v3 = {br0*p + (bc0+1): 1, (br0+1)*p + (bc0+1): -1}
        w3 = {(cc0+1)*n + cr0: 1}
        terms.append((u3, v3, w3))
        
        # P4 = a11 * (b10-b00)
        u4 = {(cr0+1)*m + (ac0+1): 1}
        v4 = {(br0+1)*p + bc0: 1, br0*p + bc0: -1}
        w4 = {(cc0)*n + cr0: 1, (cc0)*n + (cr0+1): 1}
        terms.append((u4, v4, w4))
        
        # P5 = (a00+a01)*b11
        u5 = {(cr0)*m + ac0: 1, (cr0)*m + (ac0+1): 1}
        v5 = {(br0+1)*p + (bc0+1): 1}
        w5 = {(cc0)*n + cr0: -1, (cc0+1)*n + cr0: 1}
        terms.append((u5, v5, w5))
        
        # P6 = (a10-a00)*(b00+b01)
        u6 = {(cr0+1)*m + ac0: 1, (cr0)*m + ac0: -1}
        v6 = {br0*p + bc0: 1, br0*p + (bc0+1): 1}
        w6 = {(cc0+1)*n + (cr0+1): 1}
        terms.append((u6, v6, w6))
        
        # P7 = (a01-a11)*(b10+b11)
        u7 = {(cr0)*m + (ac0+1): 1, (cr0+1)*m + (ac0+1): -1}
        v7 = {(br0+1)*p + bc0: 1, (br0+1)*p + (bc0+1): 1}
        w7 = {(cc0)*n + cr0: 1}
        terms.append((u7, v7, w7))
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U_mat[idx, base_col] = val
            for idx, val in v_dict.items():
                V_mat[idx, base_col] = val
            for idx, val in w_dict.items():
                W_mat[idx, base_col] = val
            base_col += 1
        
        return base_col

    # Schoolbook for 2x2 * 2x1 (rank 4)
    def schoolbook_2x2x1(U_mat, V_mat, W_mat, base_col, ac0, br0, bc0, cr0, cc0):
        """
        A is 2x2 (rows cr0-cr0+1, cols ac0-ac0+1)
        B is 2x1 (rows br0-br0+1, col bc0)
        C is 2x1 (rows cr0-cr0+1, col cc0)
        """
        for i in range(2):
            for j in range(2):
                u_idx = (cr0 + i) * m + (ac0 + j)
                v_idx = (br0 + j) * p + bc0
                w_idx = cc0 * n + (cr0 + i)
                U_mat[u_idx, base_col] = 1
                V_mat[v_idx, base_col] = 1
                W_mat[w_idx, base_col] = 1
                base_col += 1
        return base_col

    # Build Strassen-based decomposition
    # C = A_L * B_T + A_R * B_B
    # A_L: 2x2 (rows 0-1, cols 0-1), A_R: 2x2 (rows 0-1, cols 2-3)
    # B_T: 2x7 (rows 0-1, cols 0-6), B_B: 2x7 (rows 2-3, cols 0-6)
    
    # For A_L * B_T: B_T has 7 columns, split into 3 blocks of 2 + 1
    # Block 1: cols 0-1 (2x2) -> Strassen, rank 7
    # Block 2: cols 2-3 (2x2) -> Strassen, rank 7
    # Block 3: cols 4-5 (2x2) -> Strassen, rank 7
    # Block 4: col 6 (2x1) -> schoolbook, rank 4
    # Total: 25
    
    # For A_R * B_B: same structure, another 25
    # Total: 50
    
    U_strassen = np.zeros((dim_u, 50))
    V_strassen = np.zeros((dim_v, 50))
    W_strassen = np.zeros((dim_w, 50))
    
    col = 0
    
    # A_L * B_T: ac0=0, br0=0, cr0=0
    # B_T cols 0-1: bc0=0, cc0=0
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 0, 0, 0, 0, 0)
    # B_T cols 2-3: bc0=2, cc0=2
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 0, 0, 2, 2, 0)
    # B_T cols 4-5: bc0=4, cc0=4
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 0, 0, 4, 4, 0)
    # B_T col 6: bc0=6, cc0=6
    col = schoolbook_2x2x1(U_strassen, V_strassen, W_strassen, col, 0, 0, 6, 0, 6)
    
    # A_R * B_B: ac0=2, br0=2, cr0=0
    # B_B cols 0-1: bc0=0, cc0=0
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 2, 2, 0, 0, 0)
    # B_B cols 2-3: bc0=2, cc0=2
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 2, 2, 2, 2, 0)
    # B_B cols 4-5: bc0=4, cc0=4
    col = strassen_2x2x2(U_strassen, V_strassen, W_strassen, col, 2, 2, 4, 4, 0)
    # B_B col 6: bc0=6, cc0=6
    col = schoolbook_2x2x1(U_strassen, V_strassen, W_strassen, col, 2, 2, 6, 0, 6)
    
    # Verify
    if verify_decomposition(U_strassen, V_strassen, W_strassen, T_flat):
        best_rank = col
        best_U = U_strassen[:dim_u, :best_rank].copy()
        best_V = V_strassen[:dim_v, :best_rank].copy()
        best_W = W_strassen[:dim_w, :best_rank].copy()
    else:
        # Fallback to schoolbook
        best_rank = rank_school
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
    
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
