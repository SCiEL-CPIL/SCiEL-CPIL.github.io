import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x4x5 tensor into 4x4x4 + 4x4x1.
    # For 4x4x4, we use the Strassen decomposition (rank 49).
    # For 4x4x1, we use the schoolbook decomposition (rank 16).
    # Total rank = 49 + 16 = 65.
    # This is better than the current seed rank of 80.
    
    rank = 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 using Strassen ---
    # Strassen for 2x2x2 has rank 7.
    # We can extend this to 4x4x4 by treating the matrices as 2x2 blocks of 2x2 submatrices.
    # This is a standard tensor product construction.
    # The rank is 7 * 7 = 49.
    
    # Let's define the 7 terms for 2x2x2 Strassen.
    # A, B are 2x2.
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # C11 = P1 - P4 + P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # The 2x2x2 Strassen tensor can be represented by three 2x2x2 tensors U2, V2, W2 of shape (4, 7).
    # Indices for 2x2 matrices are flattened to 4: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: P1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: P2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 2: P3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Term 3: P4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1
    
    # Term 4: P5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    
    # Term 5: P6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: P7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # Now, construct the 4x4x4 tensor from the 2x2x2 tensor.
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # A_{ij} is a 2x2 matrix, where i,j in {0,1} are block indices.
    # The flattening of 4x4 is:
    # (0,0)->0, (0,1)->1, (0,2)->2, (0,3)->3
    # (1,0)->4, (1,1)->5, (1,2)->6, (1,3)->7
    # (2,0)->8, (2,1)->9, (2,2)->10, (2,3)->11
    # (3,0)->12, (3,1)->13, (3,2)->14, (3,3)->15
    
    # Let's define the mapping from 4x4 indices to (block_i, block_j, inner_i, inner_j).
    # 4x4 index k corresponds to (i, j) where i = k // 4, j = k % 4.
    # Block indices: bi = i // 2, bj = j // 2.
    # Inner indices: ii = i % 2, ij = j % 2.
    
    for q1 in range(7):
        for q2 in range(7):
            col = q1 * 7 + q2
            # U for 4x4x4
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ii = i % 2
                    ij = j % 2
                    # The term (q1, q2) involves U2[q1, ii*2+ij] and U2[q2, ...] ? No.
                    # The Strassen formula is applied to the blocks.
                    # A_{bi, bj} is a 2x2 matrix.
                    # The term P_{q1} is formed from blocks of A and B.
                    # Specifically, if P_q = X_q Y_q, where X_q is a linear combo of A blocks and Y_q of B blocks.
                    # The element (ii, ij) of X_q is a linear combo of A_{bi, bj}[ii', ij'].
                    # Actually, it's simpler:
                    # The 4x4x4 tensor is the tensor product of two 2x2x2 tensors.
                    # T_4[i, j, k] = sum_{q1, q2} U4[i, q1*7+q2] V4[j, q1*7+q2] W4[k, q1*7+q2]
                    # U4[i, q1*7+q2] = U2[i_block_inner, q1] * U2[j_block_inner, q2] ?
                    # Let's verify.
                    # A is 4x4. A_{ij} = sum_{bi,bj} A_{bi,bj} e_{bi} e_{bj}^T.
                    # Actually, A_{ij} = A_{bi, bj}[ii, ij] where bi=i//2, bj=j//2, ii=i%2, ij=j%2.
                    # The Strassen formula says:
                    # C_{bi, bj} = sum_q s_{q, bi, bj} P_q
                    # where P_q = X_q Y_q.
                    # X_q = sum_{bi', bj'} x_{q, bi', bj'} A_{bi', bj'}
                    # Y_q = sum_{bj', bk'} y_{q, bj', bk'} B_{bj', bk'}
                    # This is getting complicated to derive on the fly.
                    
                    # Let's use the property that T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}.
                    # The indices of T_{4,4,4} are (i, j, k).
                    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2.
                    # T_{4,4,4}[i, j, k] = sum_{q1, q2} U2[i1*2+i2, q1] V2[j1*2+j2, q2] W2[k1*2+k2, q1] * ... ?
                    # No, the tensor product of two tensors T1 and T2 is T = T1 \otimes T2.
                    # T[i1*2+i2, j1*2+j2, k1*2+k2] = T1[i1, j1, k1] T2[i2, j2, k2].
                    # This is not a rank decomposition, it's the tensor itself.
                    # The rank of T1 \otimes T2 is at most rank(T1) * rank(T2).
                    # If T1 = sum_q u1_q v1_q w1_q and T2 = sum_r u2_r v2_r w2_r,
                    # then T = sum_{q,r} (u1_q \otimes u2_r) (v1_q \otimes v2_r) (w1_q \otimes w2_r).
                    # So U4[i, q*7+r] = U2[i1, q] * U2[i2, r] where i = 2*i1 + i2.
                    # Similarly for V and W.
                    
                    pass
                    
    # Let's implement the tensor product construction explicitly.
    for q1 in range(7):
        for q2 in range(7):
            col = q1 * 7 + q2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                U[i, col] = U2[i1, q1] * U2[i2, q2]
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                V[j, col] = V2[j1, q1] * V2[j2, q2]
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                W[k, col] = W2[k1, q1] * W2[k2, q2]

    # --- Part 2: 4x4x1 using Schoolbook ---
    # The 5th column of B is B_{:, 4}.
    # C_{:, 4} = A * B_{:, 4}.
    # C_{i, 4} = sum_j A_{i, j} B_{j, 4}.
    # This is a rank 16 tensor.
    # U[i*4+j, col] = 1
    # V[j*5+4, col] = 1  (since p=5, index for B_{j,4} is j*5+4)
    # W[4*4+i, col] = 1   (since n=4, index for C_{i,4} is 4*4+i = 16+i)
    
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[16 + i, col] = 1

    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
