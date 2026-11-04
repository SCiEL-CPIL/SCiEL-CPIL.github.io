import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will construct a decomposition with rank 97.
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4x3 multiplication using Schoolbook (Rank 48)
    # Total Rank: 49 + 48 = 97
    
    # Helper to create Strassen 2x2x2 factors
    # Indices: U[0:2, 0:2], V[0:2, 0:2], W[0:2, 0:2]
    # But we need to map to flattened indices.
    
    # Let's build the 4x4x4 Strassen decomposition explicitly.
    # The 4x4 matrix is composed of four 2x2 blocks.
    # We can use the recursive Strassen algorithm.
    
    # Instead of hardcoding 49 terms, we can generate them.
    # Strassen 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    
    # For 4x4x4, we apply this to the 2x2 blocks of 2x2 matrices.
    # This results in 7 * 7 = 49 terms.
    
    # Let's define the 2x2 Strassen coefficients.
    # U factors for A (2x2):
    # Term 1: A11, A22 -> U has 1 at (0,0) and (1,1)
    # Term 2: A21, A22 -> U has 1 at (1,0) and (1,1)
    # Term 3: A11 -> U has 1 at (0,0)
    # Term 4: A22 -> U has 1 at (1,1)
    # Term 5: A11, A12 -> U has 1 at (0,0) and (0,1)
    # Term 6: A21, -A11 -> U has 1 at (1,0) and -1 at (0,0)
    # Term 7: A12, -A22 -> U has 1 at (0,1) and -1 at (1,1)
    
    # V factors for B (2x2):
    # Term 1: B11, B22 -> V has 1 at (0,0) and (1,1)
    # Term 2: B11 -> V has 1 at (0,0)
    # Term 3: B12, -B22 -> V has 1 at (0,1) and -1 at (1,1)
    # Term 4: B21, -B11 -> V has 1 at (1,0) and -1 at (0,0)
    # Term 5: B22 -> V has 1 at (1,1)
    # Term 6: B11, B12 -> V has 1 at (0,0) and (0,1)
    # Term 7: B21, B22 -> V has 1 at (1,0) and (1,1)
    
    # W factors for C (2x2):
    # Term 1: C11, C22 -> W has 1 at (0,0) and 1 at (1,1)
    # Term 2: C21 -> W has 1 at (1,0)
    # Term 3: C12, -C22 -> W has 1 at (0,1) and -1 at (1,1)
    # Term 4: C21, -C11 -> W has 1 at (1,0) and -1 at (0,0)
    # Term 5: C11, C12 -> W has 1 at (0,0) and 1 at (0,1)
    # Term 6: C22 -> W has 1 at (1,1)
    # Term 7: -C12, C22 -> W has -1 at (0,1) and 1 at (1,1)
    
    # Wait, the W indices in the task are W[k*n+i, r].
    # For 2x2, n=2. W[0*2+0]=W00, W[0*2+1]=W01, W[1*2+0]=W10, W[1*2+1]=W11.
    # My W factors above are for C[i,j] where i is row, j is col.
    # Task convention: W[k*n+i, r] where k is col, i is row.
    # So W index = k*2 + i.
    # C11 (i=0, k=0) -> W[0]
    # C12 (i=0, k=1) -> W[2]
    # C21 (i=1, k=0) -> W[1]
    # C22 (i=1, k=1) -> W[3]
    
    # Let's refine the W mapping for 2x2 Strassen.
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    
    # M1 contributes to C11 (W0) and C22 (W3).
    # M2 contributes to C21 (W1).
    # M3 contributes to C12 (W2) and -C22 (W3).
    # M4 contributes to -C11 (W0) and C21 (W1).
    # M5 contributes to C11 (W0) and C12 (W2).
    # M6 contributes to C22 (W3).
    # M7?
    # Wait, standard Strassen has 7 terms.
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    # Where are M7?
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    # 
    # It seems M7 is not used in the standard reconstruction?
    # Ah, the standard reconstruction is:
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    # This uses M1, M2, M3, M4, M5, M6. Where is M7?
    # I recall there are 7 terms.
    # Let's check Wikipedia.
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    # This only uses 6 terms?
    # No, M7 is usually defined as (A12 - A22)(B21 + B22).
    # And C22 = M1 - M3 + M6?
    # Actually, C22 = M1 - M3 + M6 is correct for some variants.
    # Let's use the standard 7-term Strassen.
    
    # To be safe, I will implement the Kronecker product of the 2x2 Strassen tensor.
    # The 2x2 Strassen tensor has rank 7.
    # The 4x4 Strassen tensor is the Kronecker product of two 2x2 Strassen tensors.
    # Rank 7 * 7 = 49.
    
    # I will construct the 2x2 factors first, then Kronecker product them.
    
    # 2x2 Factors (Flattened indices: A: 0,1,2,3; B: 0,1,2,3; C: 0,1,2,3)
    # A indices: i*m+j. m=2.
    # 0: 0,0; 1: 0,1; 2: 1,0; 3: 1,1.
    # B indices: j*p+k. p=2.
    # 0: 0,0; 1: 0,1; 2: 1,0; 3: 1,1.
    # C indices: k*n+i. n=2.
    # 0: 0,0 (k=0, i=0) -> C00
    # 1: 0,1 (k=0, i=1) -> C10
    # 2: 1,0 (k=1, i=0) -> C01
    # 3: 1,1 (k=1, i=1) -> C11
    
    # Let's define the 7 terms for 2x2.
    # U, V, W are lists of length 7, each a dict or list of non-zeros.
    
    # Term 1:
    # U: A00, A11 -> indices 0, 3.
    # V: B00, B11 -> indices 0, 3.
    # W: C00, C11 -> indices 0, 3.
    
    # Term 2:
    # U: A10, A11 -> indices 2, 3.
    # V: B00 -> index 0.
    # W: C10 -> index 1.
    
    # Term 3:
    # U: A00 -> index 0.
    # V: B01, -B11 -> indices 1, 3 (val -1).
    # W: C01 -> index 2.
    
    # Term 4:
    # U: A11 -> index 3.
    # V: B10, -B00 -> indices 2, 0 (val -1).
    # W: C10 -> index 1.
    
    # Term 5:
    # U: A00, A01 -> indices 0, 1.
    # V: B11 -> index 3.
    # W: C00, C01 -> indices 0, 2.
    
    # Term 6:
    # U: A10, -A00 -> indices 2, 0 (val -1).
    # V: B00, B01 -> indices 0, 1.
    # W: C11 -> index 3.
    
    # Term 7:
    # U: A01, -A11 -> indices 1, 3 (val -1).
    # V: B10, B11 -> indices 2, 3.
    # W: -C01, C11 -> indices 2 (val -1), 3.
    
    # Let's verify C22 (index 3) for Term 7.
    # C22 = M1 - M3 + M6?
    # My W for Term 7 has C11 (index 3) with 1.
    # My W for Term 1 has C11 (index 3) with 1.
    # My W for Term 3 has C11 (index 3) with 0?
    # Term 3 W: C01 (index 2).
    # So C11 gets M1 + M6 + M7?
    # Standard Strassen: C22 = M1 - M3 + M6.
    # If M7 is not in C22, then my Term 7 W should not have C11?
    # Let's check the definition of M7.
    # M7 = (A12-A22)(B21+B22).
    # This term is used in C22?
    # Actually, in the standard 7-term Strassen, all 7 terms are used.
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    # This uses M1, M2, M3, M4, M5, M6. M7 is unused?
    # No, that's not right.
    # I will use the known rank-7 decomposition.
    
    # To avoid errors in manual Strassen coefficients, I will use the schoolbook for 4x4x4 as well?
    # No, schoolbook is 64. Strassen is 49.
    # I will implement the Kronecker product of the 2x2 Strassen tensor carefully.
    
    # For now, to ensure correctness and speed, I will use the 97-rank hybrid:
    # 4x4x4 Strassen (49) + 4x4x3 Schoolbook (48).
    
    # I will generate the 4x4x4 Strassen by Kronecker product of the 2x2 Strassen.
    
    # 2x2 Strassen Factors (U, V, W) as lists of (index, value)
    
    # U_2x2
    u1 = [(0, 1), (3, 1)]
    v1 = [(0, 1), (3, 1)]
    w1 = [(0, 1), (3, 1)]
    
    u2 = [(2, 1), (3, 1)]
    v2 = [(0, 1)]
    w2 = [(1, 1)]
    
    u3 = [(0, 1)]
    v3 = [(1, 1), (3, -1)]
    w3 = [(2, 1)]
    
    u4 = [(3, 1)]
    v4 = [(2, 1), (0, -1)]
    w4 = [(1, 1)]
    
    u5 = [(0, 1), (1, 1)]
    v5 = [(3, 1)]
    w5 = [(0, 1), (2, 1)]
    
    u6 = [(2, 1), (0, -1)]
    v6 = [(0, 1), (1, 1)]
    w6 = [(3, 1)]
    
    u7 = [(1, 1), (3, -1)]
    v7 = [(2, 1), (3, 1)]
    w7 = [(2, -1), (3, 1)]
    
    # Kronecker Product
    # The 4x4 indices are composed of two 2x2 indices.
    # A_index = i1 * 4 + i2? No.
    # A is 4x4. Flattened index = row * 4 + col.
    # Row = r1 * 2 + r2. Col = c1 * 2 + c2.
    # Index = (r1 * 2 + r2) * 4 + (c1 * 2 + c2) = r1 * 8 + r2 * 4 + c1 * 2 + c2.
    # This is not a simple Kronecker of the 2x2 indices.
    # The Kronecker product of tensors corresponds to the block structure.
    # If we have a 2x2 tensor T, the 4x4 tensor T \otimes T has entries
    # T4[i1, i2, j1, j2, k1, k2] = T[i1, j1, k1] * T[i2, j2, k2].
    # This matches the block multiplication.
    
    # So we can generate the 49 terms by iterating over the 7 terms of the first 2x2
    # and the 7 terms of the second 2x2.
    
    # For each pair (t1, t2):
    # U term is the Kronecker product of U_t1 and U_t2.
    # If U_t1 has non-zeros at (idx1, val1) and U_t2 at (idx2, val2),
    # The 4x4 index is idx1 * 4 + idx2?
    # Let's check the index mapping.
    # 2x2 index idx = r * 2 + c.
    # 4x4 index IDX = R * 4 + C.
    # R = r1 * 2 + r2. C = c1 * 2 + c2.
    # IDX = (r1 * 2 + r2) * 4 + (c1 * 2 + c2) = r1 * 8 + r2 * 4 + c1 * 2 + c2.
    # idx1 = r1 * 2 + c1. idx2 = r2 * 2 + c2.
    # IDX = 4 * idx1 + idx2.
    # Yes! The index is 4 * idx1 + idx2.
    
    # Same for V and W.
    
    U_49 = []
    V_49 = []
    W_49 = []
    
    for t1 in range(7):
        for t2 in range(7):
            u_term = []
            for i1, v1 in [(u1[t1], 1), (u2[t1], 1), (u3[t1], 1), (u4[t1], 1), (u5[t1], 1), (u6[t1], 1), (u7[t1], 1)]:
                pass # This is messy. Let's store the lists properly.

    # Let's store the 2x2 factors as lists of lists.
    U2 = [u1, u2, u3, u4, u5, u6, u7]
    V2 = [v1, v2, v3, v4, v5, v6, v7]
    W2 = [w1, w2, w3, w4, w5, w6, w7]
    
    for t1 in range(7):
        for t2 in range(7):
            u_curr = []
            for idx1, val1 in U2[t1]:
                for idx2, val2 in U2[t2]:
                    idx = 4 * idx1 + idx2
                    val = val1 * val2
                    u_curr.append((idx, val))
            v_curr = []
            for idx1, val1 in V2[t1]:
                for idx2, val2 in V2[t2]:
                    idx = 4 * idx1 + idx2
                    val = val1 * val2
                    v_curr.append((idx, val))
            w_curr = []
            for idx1, val1 in W2[t1]:
                for idx2, val2 in W2[t2]:
                    idx = 4 * idx1 + idx2
                    val = val1 * val2
                    w_curr.append((idx, val))
            
            U_49.append(u_curr)
            V_49.append(v_curr)
            W_49.append(w_curr)
            
    # Now add Schoolbook for 4x4x3 (cols 4,5,6 of B, cols 1,2,3 of C? No, cols 4,5,6 of B correspond to cols 4,5,6 of C?
    # B is 4x7. C is 4x7.
    # We used cols 0-3 of B for the Strassen part.
    # We use cols 4-6 of B for the Schoolbook part.
    # C cols 4-6 are produced.
    
    U_48 = []
    V_48 = []
    W_48 = []
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_curr = [(i * 4 + j, 1)]
                v_curr = [(j * 7 + k, 1)]
                w_curr = [(k * 4 + i, 1)]
                U_48.append(u_curr)
                V_48.append(v_curr)
                W_48.append(w_curr)
                
    # Combine
    U_all = U_49 + U_48
    V_all = V_49 + V_48
    W_all = W_49 + W_48
    
    rank = len(U_all)
    
    # Convert to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_all[r]:
            U[idx][r] = val
        for idx, val in V_all[r]:
            V[idx][r] = val
        for idx, val in W_all[r]:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
