import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = 4, 4, 6
    rank_target = 77  # 7 outer Strassen steps × 11 inner terms each
    
    # U shape: (n*m, rank) = (16, 77)
    # V shape: (m*p, rank) = (24, 77)
    # W shape: (p*n, rank) = (24, 77)
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    
    # Block structure:
    # A is 4x4, split into 2x2 blocks of size 2x2.
    #   Block (a_row, a_col): rows [2*a_row, 2*a_row+2), cols [2*a_col, 2*a_col+2)
    # B is 4x6, split into 2x2 blocks of size 2x3.
    #   Block (b_row, b_col): rows [2*b_row, 2*b_row+2), cols [3*b_col, 3*b_col+3)
    # C is 4x6, split into 2x2 blocks of size 2x3.
    #   Block (c_row, c_col): rows [2*c_row, 2*c_row+2), cols [3*c_col, 3*c_col+3)
    
    # Outer Strassen steps for 2x2 block multiplication:
    # Each step: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_id, coeff) where block_id = a_row*2 + a_col
    # B_blocks: list of (block_id, coeff) where block_id = b_row*2 + b_col
    # C_blocks: list of (block_id, coeff) where block_id = c_row*2 + c_col
    
    outer_steps = [
        # M1 = (A00+A11)(B00+B11), C00 += M1, C11 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00, C10 += M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3 = A00(B10-B00), C01 += M3
        ([(0, 1)], [(2, 1), (0, -1)], [(1, 1)]),
        # M4 = A11(B01-B11), C11 += M4
        ([(3, 1)], [(1, 1), (3, -1)], [(3, 1)]),
        # M5 = (A00+A01)B11, C01 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1)]),
        # M6 = (A10-A00)(B00+B01), C00 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1)]),
        # M7 = (A01-A11)(B10+B11), C11 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(3, 1)]),
    ]
    
    # Inner decomposition for 2x2 * 2x3 multiplication:
    # Strassen for first 2 columns (rank 7) + schoolbook for last 2 columns (rank 4)
    # Total inner rank = 11
    
    # Strassen for 2x2 * 2x2 (first 2 columns of the 2x3 result):
    # Each term: (A_terms, B_terms, C_terms)
    # A_terms: list of (row, col, coeff) within 2x2 block
    # B_terms: list of (row, col, coeff) within 2x2 block (first 2 cols of 2x3)
    # C_terms: list of (row, col, coeff) within 2x2 block (first 2 cols of 2x3 result)
    
    inner_strassen = [
        # S1 = (A00+A11)(B00+B11), C00 += S1, C11 += S1
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # S2 = (A10+A11)B00, C10 += S2
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1)]),
        # S3 = A00(B10-B00), C01 += S3
        ([(0, 0, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 1, 1)]),
        # S4 = A11(B01-B11), C11 += S4
        ([(1, 1, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 1, 1)]),
        # S5 = (A00+A01)B11, C01 += S5
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1)]),
        # S6 = (A10-A00)(B00+B01), C00 += S6
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(0, 0, 1)]),
        # S7 = (A01-A11)(B10+B11), C11 += S7
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(1, 1, 1)]),
    ]
    
    # Schoolbook for last 2 columns of 2x2 * 2x3:
    # 4 terms for A_ij * B_jk where k is in {2, 3} (0-indexed, so cols 2 and 3 of the 2x3)
    # Actually, B is 2x3, so columns are 0, 1, 2. The "last 2 columns" are cols 1 and 2.
    # Wait, let me reconsider. The Strassen handles cols 0 and 1 of the 2x3 result.
    # Schoolbook handles cols 2 of the 2x3 result? No, that's only 1 column.
    # 
    # Let me reconsider the split. 2x2 * 2x3:
    # Result is 2x3. 
    # Option 1: Strassen for 2x2 * 2x2 (first 2 cols) + schoolbook for 2x2 * 2x1 (last col)
    #   Rank = 7 + 4 = 11
    # Option 2: Schoolbook for all 3 cols = 12
    # Option 3: Some hybrid
    #
    # Let's go with Option 1: Strassen for first 2 cols, schoolbook for last col.
    # Schoolbook for 2x2 * 2x1: 4 terms (A00*B0, A01*B1, A10*B0, A11*B1)
    
    # Actually wait - the 2x3 matrix B has 3 columns. If I use Strassen on the first 2 columns
    # (treating it as 2x2), that gives rank 7. Then schoolbook on the 3rd column (2x2 * 2x1)
    # gives rank 4. Total = 11. That's correct.
    
    # Schoolbook for 2x2 * 2x1 (just the 3rd column of B, index 2):
    inner_schoolbook = [
        # A00 * B02 -> C02
        ([(0, 0, 1)], [(0, 2, 1)], [(0, 2, 1)]),
        # A01 * B12 -> C02
        ([(0, 1, 1)], [(1, 2, 1)], [(0, 2, 1)]),
        # A10 * B02 -> C12
        ([(1, 0, 1)], [(0, 2, 1)], [(1, 2, 1)]),
        # A11 * B12 -> C12
        ([(1, 1, 1)], [(1, 2, 1)], [(1, 2, 1)]),
    ]
    
    # Helper functions to get global indices
    def get_global_A(block_id, r, s):
        # block_id = a_row * 2 + a_col
        a_row = block_id // 2
        a_col = block_id % 2
        I = a_row * 2 + r
        J = a_col * 2 + s
        return I, J  # Row I, Col J in A (4x4)
    
    def get_global_B(block_id, s, k):
        # block_id = b_row * 2 + b_col
        b_row = block_id // 2
        b_col = block_id % 2
        J = b_row * 2 + s
        K = b_col * 3 + k
        return J, K  # Row J, Col K in B (4x6)
    
    def get_global_C(block_id, r, k):
        # block_id = c_row * 2 + c_col
        c_row = block_id // 2
        c_col = block_id % 2
        I = c_row * 2 + r
        K = c_col * 3 + k
        return I, K  # Row I, Col K in C (4x6)
    
    # Build the decomposition
    for outer_idx, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = outer_idx * 11
        
        # Strassen inner terms (first 2 columns of 2x3 result)
        for inner_idx, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            c_global_rank = base_rank + inner_idx
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
            
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
            
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff
        
        # Schoolbook inner terms (3rd column of 2x3 result)
        for inner_idx, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            c_global_rank = base_rank + 7 + inner_idx
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
            
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
            
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(U, V, W, n, m, p, num_tests=5):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        for _ in range(num_tests):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            
            # Compute P[k] = sum_{i,j} U[i,j,k] * A[i,j]
            M = np.einsum('ijk,ij->k', U_r, A)
            # Compute Q[k] = sum_{j,l} V[j,l,k] * B[j,l]
            N = np.einsum('jlk,jl->k', V_r, B)
            
            P = M * N
            # Compute C[i,k] = sum_t W[k,i,t] * P[t]
            C_comp = np.zeros((n, p))
            for I in range(n):
                for K in range(p):
                    C_comp[I, K] = np.dot(P, W_r[K, I])
            
            C_ref = A @ B
            if not np.allclose(C_comp, C_ref, atol=1e-6):
                return False
        return True
    
    is_valid = verify_decomp(U_arr, V_arr, W_arr, n, m, p, num_tests=5)
    
    if not is_valid:
        # Fallback: schoolbook decomposition (rank = n*m*p = 96)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_f[i * m + j][idx] = 1
                    V_f[j * p + k][idx] = 1
                    W_f[k * n + i][idx] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
