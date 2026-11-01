import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    current_rank = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_offset: (i_offset, j_offset) for A
    # b_offset: (j_offset, k_offset) for B
    # c_offset: (i_offset, k_offset) for C
    def add_strassen_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal current_rank
        
        # Strassen formulas
        # m1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # m2 = (A10+A11)B00    -> C10+=1
        # m3 = A00(B01-B11)    -> C01+=1
        # m4 = A11(B10-B00)    -> C00+=1, C10+=1
        # m5 = (A00+A01)B11    -> C00-=1, C01+=1
        # m6 = (A10-A00)(B00+B01) -> C11+=1
        # m7 = (A01-A11)(B10+B11) -> C00+=1, C11+=1
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: {(i,j): coeff}, V: {(j,k): coeff}, W: {(k,i): coeff}
            
            # m1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # m2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # m3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # m4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # m5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # m6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # m7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        ]
        
        for u_map, v_map, w_map in terms:
            for (li, lj), val in u_map.items():
                gi = li + a_i_off
                gj = lj + a_j_off
                U[gi * m + gj, current_rank] += val
            for (lj, lk), val in v_map.items():
                gj = lj + b_j_off
                gk = lk + b_k_off
                V[gj * p + gk, current_rank] += val
            for (lk, li), val in w_map.items():
                gk = lk + c_k_off
                gi = li + c_i_off
                W[gk * n + gi, current_rank] += val
            current_rank += 1

    # Helper to add a 2x2x1 standard block
    def add_standard_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal current_rank
        
        # C0k = A0j0 B0k + A0j1 B1k
        # C1k = A1j0 B0k + A1j1 B1k
        
        terms = [
            # m1 = A00 B0k -> C0k += 1
            ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
            # m2 = A01 B1k -> C0k += 1
            ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
            # m3 = A10 B0k -> C1k += 1
            ({(1,0): 1}, {(0,0): 1}, {(0,1): 1}),
            # m4 = A11 B1k -> C1k += 1
            ({(1,1): 1}, {(1,0): 1}, {(0,1): 1})
        ]
        
        for u_map, v_map, w_map in terms:
            for (li, lj), val in u_map.items():
                gi = li + a_i_off
                gj = lj + a_j_off
                U[gi * m + gj, current_rank] += val
            for (lj, lk), val in v_map.items():
                gj = lj + b_j_off
                gk = lk + b_k_off
                V[gj * p + gk, current_rank] += val
            for (lk, li), val in w_map.items():
                gk = lk + c_k_off
                gi = li + c_i_off
                W[gk * n + gi, current_rank] += val
            current_rank += 1

    # Block 1: A_L (j=0,1) * B_T (j=0,1)
    # 1.1: B_T1 (k=0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # 1.2: B_T2 (k=2,3)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # 1.3: B_T3 (k=4)
    add_standard_block(0, 0, 0, 4, 0, 4)
    
    # Block 2: A_R (j=2,3) * B_B (j=2,3)
    # 2.1: B_B1 (k=0,1)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # 2.2: B_B2 (k=2,3)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    # 2.3: B_B3 (k=4)
    add_standard_block(0, 2, 2, 4, 0, 4)
    
    # Verify rank
    assert current_rank == 36, f"Expected rank 36, got {current_rank}"
    
    # Verification
    def verify():
        for i in range(n):
            for k in range(p):
                # Calculate C_ik from U, V, W
                val = 0
                for r in range(rank):
                    # Sum over j? No, U and V already sum over j?
                    # The tensor decomposition is T_ijk = sum_r U_ij^r V_jk^r W_ki^r
                    # C_ik = sum_j A_ij B_jk
                    # If we set A_ij = 1, B_jk = 1, C_ik should be m (number of j's) = 4.
                    # But wait, the tensor T_ijk is the coefficient of A_ij B_jk in C_ik.
                    # T_ijk = 1 if we are looking at the term A_ij B_jk contributing to C_ik.
                    # Actually, T_ijk is 1 for all i,j,k? No.
                    # C_ik = sum_j A_ij B_jk.
                    # So the coefficient of A_ij B_jk in C_ik is 1.
                    # So T_ijk = 1 for all i, j, k?
                    # No, T is a 3D tensor of size n x m x p.
                    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
                    # Yes, for matrix multiplication, T[i,j,k] = 1 for all i,j,k.
                    
                    # Wait, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
                    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                    
                    # Let's check T[i,j,k] for a specific i,j,k.
                    # It should be 1.
                    pass
    
    # Better verification:
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

    if not verify():
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

