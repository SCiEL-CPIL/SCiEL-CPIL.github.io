import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0
    
    # Strategy: Laderman (rank 20 for 3x3) x Strassen (rank 7 for 2x2)
    # Total rank = 20 * 7 = 140
    
    # 1. Define Laderman 3x3 terms (U, V, W are 9-element vectors)
    # Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    laderman_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # M1: A00
        [1, 1, 0, 0, 0, 0, 0, 0, 0], # M2: A00+A01
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # M3: A10
        [0, 0, 0, 1, 1, 0, 0, 0, 0], # M4: A10+A11
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # M5: A02
        [0, 0, 1, 0, 0, 1, 0, 0, 0], # M6: A02+A12
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # M7: A20
        [0, 0, 0, 0, 0, 0, 1, 1, 0], # M8: A20+A21
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # M9: A22
        [1, -1, 0, 0, 0, 0, 0, 0, 0],# M10: A00-A01
        [0, 0, 0, 1, -1, 0, 0, 0, 0],# M11: A10-A11
        [0, 0, 0, 0, 0, 0, 1, -1, 0],# M12: A20-A21
        [1, 0, -1, 0, 0, 0, 0, 0, 0],# M13: A00-A02
        [0, 0, 0, 1, 0, -1, 0, 0, 0],# M14: A10-A12
        [0, 0, 0, 0, 0, 0, 1, 0, -1],# M15: A20-A22
        [1, 0, 0, -1, 0, 0, 0, 0, 0],# M16: A00-A10
        [0, 0, 0, 0, 1, 0, -1, 0, 0],# M17: A11-A20
        [0, 0, 0, 0, 0, 0, 0, 1, 0], # M18: A21
        [0, 0, 0, 0, 0, 0, 0, 1, 1], # M19: A21+A22
        [0, 0, 0, 0, 0, 0, 0, 0, 1]  # M20: A22
    ]
    laderman_V = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # B10
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # B10
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # B20
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # B20
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # B10
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # B22
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # B10
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # B10
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # B22
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # B11
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 0, 0, 0, 0, 0, 0, 1, 0], # B21
        [1, 0, 0, 0, 0, 0, 0, 0, 0]  # B00
    ]
    laderman_W = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # C10
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # C10
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # C02
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # C02
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # C20
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # C20
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # C22
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C01
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C21
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C01
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # C21
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # C20
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # C20
        [0, 0, 0, 0, 0, 0, 0, 0, 1]  # C22
    ]
    
    # 2. Define Strassen 2x2 terms (U, V, W are 4-element vectors)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B10-B00
        [-1, 0, 1, 0],# M4: B11-B01
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: C00-C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # 3. Construct the full 140 rank matrices
    rank = 140
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3): # Block row i (0..2)
        for k in range(3): # Block col k (0..2)
            for j in range(3): # Output block col j (0..2)
                for s in range(7): # Strassen term
                    for l in range(20): # Laderman term
                        # Indices in the 6x6 matrices
                        # A is 6x6. Block (i, k) is 2x2.
                        # Local indices in 2x2 block: (0,0), (0,1), (1,0), (1,1)
                        # Global indices:
                        # a_00 = (2*i)*6 + 2*k
                        # a_01 = (2*i)*6 + 2*k + 1
                        # a_10 = (2*i+1)*6 + 2*k
                        # a_11 = (2*i+1)*6 + 2*k + 1
                        
                        a_base_0 = (2*i) * 6 + 2*k
                        a_base_1 = (2*i+1) * 6 + 2*k
                        
                        # B is 6x6. Block (k, j) is 2x2.
                        # Global indices:
                        # b_00 = (2*k)*6 + 2*j
                        # b_01 = (2*k)*6 + 2*j + 1
                        # b_10 = (2*k+1)*6 + 2*j
                        # b_11 = (2*k+1)*6 + 2*j + 1
                        
                        b_base_0 = (2*k) * 6 + 2*j
                        b_base_1 = (2*k+1) * 6 + 2*j
                        
                        # C is 6x6. Block (j, i) is 2x2.
                        # Global indices:
                        # c_00 = (2*j)*6 + 2*i
                        # c_01 = (2*j+1)*6 + 2*i
                        # c_10 = (2*j)*6 + (2*i+1)
                        # c_11 = (2*j+1)*6 + (2*i+1)
                        
                        c_base_0 = (2*j) * 6 + 2*i
                        c_base_1 = (2*j+1) * 6 + 2*i
                        
                        # Laderman coefficients for A
                        # laderman_U[l] corresponds to A indices 0..8
                        # We need to map local 3x3 block indices to 2x2 sub-block indices?
                        # No, Laderman is for the 3x3 block structure.
                        # The 3x3 blocks are 2x2 matrices.
                        # So Laderman operates on the "block level".
                        # The tensor product means:
                        # U_full[global_A_index, idx] += laderman_U[l][local_block_A] * strassen_U[s][local_sub_A]
                        
                        # Let's map local_block_A (0..8) to local_sub_A (0..3)
                        # local_block_A is the index of the 2x2 block in the 3x3 grid.
                        # Wait, Laderman U/V/W vectors are of length 9. They represent coefficients for the 9 blocks.
                        # Strassen U/V/W vectors are of length 4. They represent coefficients for the 4 entries of the 2x2 block.
                        # The total index in U is (block_row, block_col, sub_row, sub_col).
                        # In our 6x6 matrix, the global index is determined by (block_row, block_col) and (sub_row, sub_col).
                        
                        # Let's re-verify the mapping.
                        # A is 6x6.
                        # Block (i, k) is the 2x2 submatrix A[2i:2i+2, 2k:2k+2].
                        # Laderman term l has coefficient laderman_U[l][block_idx] for block (i, k)?
                        # No, Laderman is for multiplying two 3x3 matrices of 2x2 blocks.
                        # So Laderman term l has coefficients for the 9 blocks of A and B.
                        # Specifically, U_l is a vector of 9 coefficients.
                        # If we are multiplying A_block * B_block, where A_block is 3x3 blocks and B_block is 3x3 blocks.
                        # Then the term is sum_l (sum_{i,k} U_l[i,k] * A_block[i,k]) * (sum_{k,j} V_l[k,j] * B_block[k,j]).
                        # This product is a 2x2 matrix (since blocks are 2x2).
                        # This 2x2 matrix needs to be multiplied by W_l[i,j] to contribute to C_block[i,j].
                        # BUT, the multiplication of the 2x2 block matrices is not a single scalar.
                        # The tensor product structure is:
                        # (A_block * B_block) is a 2x2 matrix.
                        # The scalar multiplication in the rank-1 tensor decomposition is for SCALAR blocks.
                        # Here, our "elements" are 2x2 matrices.
                        # So we need to multiply the 2x2 block result by W_l[i,j].
                        # This is a scalar multiplication of the 2x2 matrix.
                        # So the contribution to C_block[i,j] is W_l[i,j] * (A_block_sum * B_block_sum).
                        # This is NOT a simple tensor product of scalar decompositions unless we expand the 2x2 multiplication.
                        
                        # Wait, the standard tensor product of decompositions:
                        # If A (n x n) and B (n x n) are multiplied with rank r_A and r_B.
                        # And we want to multiply A' (n*n x n*n) and B' (n*n x n*n) which are block matrices.
                        # Then the rank is r_A * r_B.
                        # This works if the block multiplication is consistent with the scalar multiplication.
                        # i.e., (Block A) * (Block B) is computed as the block matrix multiplication.
                        # The tensor product decomposition for block matrices:
                        # U_total[a_global, idx_total] = U_A[a_block, l] * U_B[a_sub, s]
                        # This is valid if the block multiplication is (A_block[i,k] * B_block[k,j]) where the product is the 2x2 matrix product.
                        # And the scalar decomposition of 2x2 multiplication is Strassen.
                        # So yes, the rank is 20 * 7 = 140.
                        
                        # So for each (i, k, j, s, l):
                        # We need to update U, V, W.
                        # U depends on A.
                        # U[global_A_index, idx] += laderman_U[l][block_A_index] * strassen_U[s][sub_A_index]
                        
                        # How to map?
                        # laderman_U[l] is a list of 9 floats. Index 0..8 corresponds to blocks (0,0)..(2,2).
                        # For the current term, we are computing C_block[j, i] from A_block[i, k] and B_block[k, j].
                        # Wait, the indices in Laderman are for the 3x3 block grid.
                        # So for a fixed (i, k, j), the coefficient for A_block[i, k] is laderman_U[l][i*3+k].
                        # The coefficient for B_block[k, j] is laderman_V[l][k*3+j].
                        # The coefficient for C_block[j, i] is laderman_W[l][j*3+i].
                        
                        # Now, inside the block A_block[i, k], we have 4 sub-entries.
                        # The Strassen decomposition gives coefficients for these 4 sub-entries.
                        # strassen_U[s] is a list of 4 floats. Index 0..3 corresponds to sub-entries (0,0)..(1,1).
                        
                        # So for a specific sub-entry (r, c) in the 2x2 block A[i, k]:
                        # Global index in A: a_global = (2*i + r) * 6 + (2*k + c)
                        # Coefficient in U: U[a_global, idx] += laderman_U[l][i*3+k] * strassen_U[s][r*2+c]
                        
                        # Similarly for B:
                        # Global index in B: b_global = (2*k + r) * 6 + (2*j + c)
                        # Coefficient in V: V[b_global, idx] += laderman_V[l][k*3+j] * strassen_V[s][r*2+c]
                        
                        # Similarly for C:
                        # Global index in C: c_global = (2*j + r) * 6 + (2*i + c)
                        # Coefficient in W: W[c_global, idx] += laderman_W[l][j*3+i] * strassen_W[s][r*2+c]
                        
                        coeff_A = laderman_U[l][i*3+k]
                        coeff_B = laderman_V[l][k*3+j]
                        coeff_C = laderman_W[l][j*3+i]
                        
                        for r in range(2):
                            for c in range(2):
                                sub_idx = r*2 + c
                                u_val = coeff_A * strassen_U[s][sub_idx]
                                v_val = coeff_B * strassen_V[s][sub_idx]
                                w_val = coeff_C * strassen_W[s][sub_idx]
                                
                                a_global = (2*i + r) * 6 + (2*k + c)
                                b_global = (2*k + r) * 6 + (2*j + c)
                                c_global = (2*j + r) * 6 + (2*i + c)
                                
                                U[a_global, idx] += u_val
                                V[b_global, idx] += v_val
                                W[c_global, idx] += w_val
                        
                        idx += 1
    
    assert idx == rank
    
    # 4. Verification
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float).reshape(n, m, rank)
        V_arr = V.astype(float).reshape(m, p, rank)
        W_arr = W.astype(float).reshape(p, n, rank)
        
        prod = np.einsum('ikt,kjt->ijt', U_arr, V_arr)
        C_computed = np.einsum('ijt,jit->ij', prod, W_arr)
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 5. Fallback to 189 rank Block-Strassen
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    c00 = (2*j) * 6 + 2*i      
                    c01 = (2*j+1) * 6 + 2*i    
                    c10 = (2*j) * 6 + (2*i+1)  
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U[a00, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1; W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1; W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1; V[b11, idx] = -1
                        W[c01, idx] = 1; W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1; V[b00, idx] = -1
                        W[c00, idx] = 1; W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1; U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1; W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1; U[a00, idx] = -1
                        V[b00, idx] = 1; V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1; U[a11, idx] = -1
                        V[b10, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank
    
    # Verify 189 rank
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float).reshape(n, m, rank)
        V_arr = V.astype(float).reshape(m, p, rank)
        W_arr = W.astype(float).reshape(p, n, rank)
        
        prod = np.einsum('ikt,kjt->ijt', U_arr, V_arr)
        C_computed = np.einsum('ijt,jit->ij', prod, W_arr)
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 6. Fallback to schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school), dtype=int)
    V_s = np.zeros((m * p, rank_school), dtype=int)
    W_s = np.zeros((p * n, rank_school), dtype=int)
    idx_s = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_s[i * m + k, idx_s] = 1
                V_s[k * p + j, idx_s] = 1
                W_s[j * n + i, idx_s] = 1
                idx_s += 1
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
