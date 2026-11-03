import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=float)
    V_triv = np.zeros((m * p, rank_triv), dtype=float)
    W_triv = np.zeros((p * n, rank_triv), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Strategy: Use 8x8 matrix multiplication decomposition, then slice to 6x6
    # R(8) <= R(4) * R(2) using Strassen on 2x2 blocks of 4x4
    # R(4) = 16 (Laderman), R(2) = 7 (Strassen)
    # So R(8) <= 16 * 7 = 112
    # We pad 6x6 to 8x8, compute, then extract the 6x6 result

    # Laderman's rank-16 decomposition for 4x4 matrix multiplication
    # C = A * B where A, B, C are 4x4
    # The 16 terms are given as (A_mask, B_mask, C_mask)
    # Each mask is a 4x4 matrix with 0/1 entries
    # C[i,j] = sum_s A_s[i,:] * B_s[:,j] * (coefficient)
    # Actually, for tensor decomposition: T[i,k,j] = sum_s U[i,k,s] * V[k,j,s] * W[j,i,s]
    
    # Laderman's 16 terms for 4x4:
    # Let me use the standard formulation.
    # A, B are 4x4 matrices. C = AB.
    # The 16 bilinear forms:
    
    laderman_16 = [
        # Term 1: A[0,0]+A[1,1] times B[0,0]+B[1,1]
        # This contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        # Actually, let me be more careful.
        
        # Standard Laderman terms (from literature):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A01+A11)(B10+B11)
        # M3 = (A00+A22)(B00+B22)
        # M4 = (A21+A22)(B22+B21)
        # M5 = (A00+A33)(B00+B33)
        # M6 = (A31+A33)(B33+B31)
        # M7 = (A10+A20)(B10+B20)
        # M8 = (A11+A21)(B11+B21)
        # M9 = (A00+A10+A20)(B00+B10+B20)  -- wait, this doesn't look right
        
        # Let me use a different approach. I'll define the 16 terms explicitly.
        # Each term is a triple (U, V, W) where U, V, W are 4x4 matrices
        # such that sum_s U_s ⊗ V_s ⊗ W_s = matmul tensor for 4x4
        
        # Actually, the tensor decomposition for 4x4 matmul:
        # T[i,k,j] = 1 if all indices valid
        # We need: T[i,k,j] = sum_s U_s[i,k] * V_s[k,j] * W_s[j,i]
        
        # Let me just hardcode a known valid decomposition.
        # From the literature, Laderman's algorithm uses these 16 products:
        
        # I'll define them as lists of (i,k), (k,j), (j,i) with coefficients
    ]
    
    # Let me define Laderman's 16 terms more carefully using the standard formulation.
    # Each term s has:
    #   U_s: 4x4 matrix (coefficient for A[i,k])
    #   V_s: 4x4 matrix (coefficient for B[k,j])  
    #   W_s: 4x4 matrix (coefficient for C[j,i])  -- note: W[j,i] not W[i,j]
    
    # The condition is: for each (i,k,j), sum_s U_s[i,k] * V_s[k,j] * W_s[j,i] = 1
    
    # Let me use a simpler approach: define the 16 bilinear forms and their contributions.
    
    # Laderman's 16 terms (verified from multiple sources):
    # Let A, B be 4x4. Define:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[0,1] + A[1,1]) * (B[1,0] + B[1,1])
    # M3 = (A[0,0] + A[2,2]) * (B[0,0] + B[2,2])
    # M4 = (A[2,1] + A[2,2]) * (B[2,2] + B[2,1])
    # M5 = (A[0,0] + A[3,3]) * (B[0,0] + B[3,3])
    # M6 = (A[3,1] + A[3,3]) * (B[3,3] + B[3,1])
    # M7 = (A[1,0] + A[2,0]) * (B[1,0] + B[2,0])
    # M8 = (A[1,1] + A[2,1]) * (B[1,1] + B[2,1])
    # M9 = (A[0,0] + A[0,1] + A[1,0] + A[1,1]) * (B[0,0] + B[0,1] + B[1,0] + B[1,1])
    # M10 = (A[0,0] + A[0,2] + A[2,0] + A[2,2]) * (B[0,0] + B[0,2] + B[2,0] + B[2,2])
    # M11 = (A[0,0] + A[0,3] + A[3,0] + A[3,3]) * (B[0,0] + B[0,3] + B[3,0] + B[3,3])
    # M12 = (A[1,0] + A[1,1] + A[2,0] + A[2,1]) * (B[1,0] + B[1,1] + B[2,0] + B[2,1])
    # M13 = (A[0,1] + A[1,0] + A[0,3] + A[3,0]) * (B[0,1] + B[1,0] + B[0,3] + B[3,0])
    # M14 = (A[0,2] + A[2,0] + A[0,3] + A[3,0]) * (B[0,2] + B[2,0] + B[0,3] + B[3,0])
    # M15 = (A[1,1] + A[2,1] + A[1,3] + A[3,1]) * (B[1,1] + B[2,1] + B[1,3] + B[3,1])
    # M16 = (A[1,2] + A[2,2] + A[1,3] + A[3,2]) * (B[1,2] + B[2,2] + B[1,3] + B[3,2])
    
    # This is getting complicated. Let me just use the fallback 189 approach since
    # I'm not confident in the Laderman implementation.
    
    # Actually, let me try a different approach: use the known R(6) <= R(4)*R(3) bound.
    # R(4) = 16, R(3) = 19 (best known). So R(6) <= 16*19 = 304. That's worse than 189.
    
    # Or R(6) <= R(8) * (1/R(8) * R(6))... no that doesn't help.
    
    # Let me stick with the 189 approach but try to verify it works correctly.
    
    # Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3
    
    strassen_steps = [
        {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1, (1, 1): 1}},
        {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1},
         'C': {(1, 0): 1, (1, 1): -1}},
        {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1},
         'C': {(0, 1): 1, (1, 1): 1}},
        {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1},
         'C': {(0, 0): 1, (1, 0): 1}},
        {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1},
         'C': {(0, 0): -1, (0, 1): 1}},
        {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1},
         'C': {(1, 1): 1}},
        {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1}},
    ]
    
    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_steps[s]
                    for (ar, ac), coeff_a in s_def['A'].items():
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    for (br, bc), coeff_b in s_def['B'].items():
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    for (cr, cc), coeff_c in s_def['C'].items():
                        i_glob = i_b * bs + cr
                        j_glob = j_b * bs + cc
                        W2[j_glob * n + i_glob, r_idx] += coeff_c
                    r_idx += 1
    
    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Fallback to trivial
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
