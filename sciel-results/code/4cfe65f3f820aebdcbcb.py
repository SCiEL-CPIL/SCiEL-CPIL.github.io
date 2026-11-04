import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Target rank 33
    rank = 33
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    
    col = 0

    # Helper to add a rank-1 term
    def add_rank1(u_idx, v_idx, w_idx, val_u=1.0, val_v=1.0, val_w=1.0):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] += val_u
        V[v_idx, col] += val_v
        W[w_idx, col] += val_w
        col += 1

    # Helper to add a linear combination of rank-1 terms for Strassen
    # This is not quite right, Strassen combines inputs before multiplication.
    # We need to expand the Strassen products into rank-1 terms of the original tensor.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # Each term Aij Bkl contributes to C.
    # The tensor decomposition of Aij Bkl is a single rank-1 term if we map indices correctly?
    # No, Aij Bkl is a product of two variables.
    # In the tensor T, T[a,b,c] = 1 if c corresponds to i,k.
    # The term Aij Bkl appears in C_ik.
    # So for a fixed i,k, the coefficient of Aij Bkl in C_ik is 1 if j is the summing index?
    # No, C_ik = sum_j Aij Bjk.
    # So Aij Bjk is a term in C_ik.
    # The tensor entry T[a,b,c] for a=i*m+j, b=j*p+k, c=k*n+i is 1.
    # This corresponds to the product A_{i,j} B_{j,k} contributing to C_{i,k}.
    # So the "rank-1" term in the tensor decomposition is exactly this triple (a,b,c).
    # Strassen's algorithm expresses C_ik as a linear combination of 7 products M_r.
    # Each M_r is a product of a linear combo of A's and a linear combo of B's.
    # M_r = (sum c_u A_u) (sum d_v B_v) = sum_{u,v} c_u d_v A_u B_v.
    # Each A_u B_v corresponds to a specific tensor entry (if u, v are compatible indices).
    # So we can expand each M_r into a sum of rank-1 tensor terms (the standard basis).
    # The rank of the decomposition is the number of non-zero coefficients in the expanded sum?
    # No, Strassen reduces the number of multiplications, but in the tensor decomposition context,
    # the "rank" is the number of terms in the sum U_r V_r W_r.
    # If we expand M_r into A_u B_v terms, we are back to schoolbook rank.
    # Wait, the Strassen algorithm works because we can share the intermediate products.
    # But the tensor decomposition rank is the minimum number of rank-1 tensors that sum to T.
    # The Strassen decomposition of the 2x2x2 tensor has rank 7.
    # This means T_2x2x2 = sum_{r=1}^7 U_r \otimes V_r \otimes W_r.
    # The U_r, V_r, W_r are vectors in the input/output spaces.
    # For 2x2x2, the spaces are dimension 4, 4, 4.
    # The 7 terms are specific linear combinations.
    
    # Let's explicitly define the 7 Strassen terms for 2x2x2.
    # Inputs A: indices 0,1,2,3 (00,01,10,11).
    # Inputs B: indices 0,1,2,3 (00,01,10,11).
    # Outputs C: indices 0,1,2,3 (00,01,10,11).
    
    # We will map the global indices to local 2x2 indices for each block.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Global A: 0,1,3,4
    # Global B: 0,1,6,7
    # Global C: 0,1,6,7
    
    # Strassen U, V, W vectors for 2x2x2 (local indices 0..3)
    # U vectors (for A):
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00+A01 -> [1, 1, 0, 0]
    # M6: A10-A00 -> [-1, 0, 1, 0]
    # M7: A01-A11 -> [0, 1, 0, -1]
    
    # V vectors (for B):
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    # M4: B10-B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    
    # W vectors (for C):
    # C00 = M1+M4-M5+M7 -> W00 has coeffs [1, 0, 0, 1, -1, 0, 1] for M1..M7?
    # No, W_r is the vector for the r-th term.
    # C = sum_r M_r W_r.
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # So W_1 (for M1) at C00 is 1.
    # W_4 (for M4) at C00 is 1.
    # W_5 (for M5) at C00 is -1.
    # W_7 (for M7) at C00 is 1.
    # Others 0.
    
    # We need to construct U, V, W global matrices.
    # For each Strassen term r in 1..7:
    # U_global[local_A_idx, r] = U_local[local_A_idx]
    # V_global[local_B_idx, r] = V_local[local_B_idx]
    # W_global[local_C_idx, r] = W_local[local_C_idx]
    
    # Local A indices: 0->00, 1->01, 2->10, 3->11
    # Global A for A[0:2, 0:2]:
    # 00 -> 0, 01 -> 1, 10 -> 3, 11 -> 4
    
    # Local B indices: 0->00, 1->01, 2->10, 3->11
    # Global B for B[0:2, 0:2]:
    # 00 -> 0, 01 -> 1, 10 -> 6, 11 -> 7
    
    # Local C indices: 0->00, 1->01, 2->10, 3->11
    # Global C for C[0:2, 0:2]:
    # 00 -> 0, 01 -> 1, 10 -> 6, 11 -> 7
    
    # Define the 7 terms
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # W_r is a vector of length 4 (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = [
        [1, 0, 0, 0], # M1 contributes to C00, C11
        [0, 0, 1, -1], # M2 contributes to C10, C11
        [0, 1, 0, 1], # M3 contributes to C01, C11
        [1, 0, 1, 0], # M4 contributes to C00, C10
        [-1, 1, 0, 0], # M5 contributes to C00, C01
        [0, 0, 0, 1], # M6 contributes to C11
        [1, 0, 0, 0]  # M7 contributes to C00
    ]
    # Correction: W_r[k] is the coefficient of M_r in C_k.
    # C00: M1(1), M4(1), M5(-1), M7(1).
    # So W_1[0]=1, W_4[0]=1, W_5[0]=-1, W_7[0]=1.
    # C01: M3(1), M5(1).
    # W_3[1]=1, W_5[1]=1.
    # C10: M2(1), M4(1).
    # W_2[2]=1, W_4[2]=1.
    # C11: M1(1), M2(-1), M3(1), M6(1).
    # W_1[3]=1, W_2[3]=-1, W_3[3]=1, W_6[3]=1.
    
    strassen_W_correct = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1], # M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0], # M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]

    # Process 3 Strassen blocks for A[0:2, 0:2] * B[0:2, 0:6]
    for block in range(3):
        # B block indices: 0:2, 2:4, 4:6
        b_offset = block * 2
        # Global B indices for this 2x2 block
        # B[0, k] -> 6*0 + k = k
        # B[1, k] -> 6*1 + k = 6+k
        # Local B 00 -> Global 0 + b_offset
        # Local B 01 -> Global 1 + b_offset
        # Local B 10 -> Global 6 + b_offset
        # Local B 11 -> Global 7 + b_offset
        
        # Global A indices for A[0:2, 0:2] are constant: 0, 1, 3, 4
        a_map = [0, 1, 3, 4]
        
        # Global C indices for C[0:2, 0:2] are constant: 0, 1, 6, 7
        # C[0, k] -> 6*k + 0
        # C[1, k] -> 6*k + 1
        # Local C 00 -> Global 0 + b_offset? No, C cols are 0..5.
        # Block 0: C cols 0,1. C[0,0]->0, C[0,1]->1, C[1,0]->6, C[1,1]->7.
        # Block 1: C cols 2,3. C[0,2]->2, C[0,3]->3, C[1,2]->12, C[1,3]->13.
        # Block 2: C cols 4,5. C[0,4]->4, C[0,5]->5, C[1,4]->18? No 6*4+1=25?
        # Wait, C is 2x6. c = 6k + i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 6
        # k=1, i=1 -> 7
        # k=2, i=0 -> 12
        # k=2, i=1 -> 13
        # k=3, i=0 -> 18? No, 6*3+0=18. But C has 12 entries (0..11).
        # Ah, p=6, n=2. c = k*n + i = 2k + i?
        # Task says: W[k*n+i, r].
        # n=2. c = 2k + i.
        # k=0..5, i=0..1.
        # c ranges 0..11.
        # Block 0 (k=0,1): c=0,1,2,3.
        # Block 1 (k=2,3): c=4,5,6,7.
        # Block 2 (k=4,5): c=8,9,10,11.
        
        c_map = [2*b_offset, 2*b_offset+1, 2*b_offset+2, 2*b_offset+3]
        # Local C 00 -> c_map[0]
        # Local C 01 -> c_map[1]
        # Local C 10 -> c_map[2]
        # Local C 11 -> c_map[3]
        
        b_map = [b_offset, b_offset+1, 6+b_offset, 6+b_offset+1]
        
        for r in range(7):
            for l_a in range(4):
                if strassen_U[r][l_a] != 0:
                    U[a_map[l_a], col] += strassen_U[r][l_a]
            for l_b in range(4):
                if strassen_V[r][l_b] != 0:
                    V[b_map[l_b], col] += strassen_V[r][l_b]
            for l_c in range(4):
                if strassen_W_correct[r][l_c] != 0:
                    W[c_map[l_c], col] += strassen_W_correct[r][l_c]
            col += 1

    # Process 3 Standard blocks for A[0:2, 2:3] * B[2:3, 0:6]
    # A[0,2] -> a=2, A[1,2] -> a=5
    # B[2,k] -> b=12+k
    # C[i,k] -> c=2k+i
    
    for block in range(3):
        b_offset = block * 2
        # B indices: 12+b_offset, 13+b_offset
        b_map = [12+b_offset, 13+b_offset]
        
        # C indices:
        # k = b_offset, b_offset+1
        # i = 0, 1
        # c = 2k + i
        # k=b_offset, i=0 -> 2*b_offset
        # k=b_offset, i=1 -> 2*b_offset + 1
        # k=b_offset+1, i=0 -> 2*b_offset + 2
        # k=b_offset+1, i=1 -> 2*b_offset + 3
        c_map = [2*b_offset, 2*b_offset+1, 2*b_offset+2, 2*b_offset+3]
        
        # A indices: 2, 5
        a_map = [2, 5]
        
        # Rank 4 decomposition for 2x1x2
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        # r=0: A0 B0 -> C00
        U[a_map[0], col] = 1
        V[b_map[0], col] = 1
        W[c_map[0], col] = 1
        col += 1
        
        # r=1: A0 B1 -> C01
        U[a_map[0], col] = 1
        V[b_map[1], col] = 1
        W[c_map[1], col] = 1
        col += 1
        
        # r=2: A1 B0 -> C10
        U[a_map[1], col] = 1
        V[b_map[0], col] = 1
        W[c_map[2], col] = 1
        col += 1
        
        # r=3: A1 B1 -> C11
        U[a_map[1], col] = 1
        V[b_map[1], col] = 1
        W[c_map[3], col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
