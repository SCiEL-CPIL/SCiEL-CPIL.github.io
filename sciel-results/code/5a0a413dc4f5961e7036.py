import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Create rank-7 decomposition for 2x2x2
    # U, V, W shapes (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[1, 3] = 1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[2, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6
    U2[0, 5] = -1; U2[1, 5] = 1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[2, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Step 2: Expand to 4x4x4 using Kronecker products and permutation
    # The tensor is T_4 = T_2 x T_2
    # We need to map indices correctly.
    # U4 row index: i4 * 4 + j4. i4 = i1*2 + i0, j4 = j1*2 + j0.
    # U2 row index: i2 * 2 + j2.
    # Kronecker U2 x U2 has row (r1 * 4 + r0) where r1 = i1*2 + j1, r0 = i0*2 + j0.
    # This results in index: (i1*2 + j1)*4 + (i0*2 + j0) = 8*i1 + 4*j1 + 2*i0 + j0.
    # Desired index: 8*i1 + 4*i0 + 2*j1 + j0.
    # We need to permute the rows of the Kronecker product.
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutation for U4 and V4 (i,j structure)
    # Map (i1, j1, i0, j0) from kron order (i1, j1, i0, j0) -> desired (i1, i0, j1, j0)
    # Kron row: i1*8 + j1*4 + i0*2 + j0
    # Desired row: i1*8 + i0*4 + j1*2 + j0
    
    perm_U4 = []
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    kron_idx = i1*8 + j1*4 + i0*2 + j0
                    desired_idx = i1*8 + i0*4 + j1*2 + j0
                    perm_U4.append((desired_idx, kron_idx))
    
    # Sort by desired_idx to create a permutation vector
    perm_map_U4 = [0]*16
    for d, k in perm_U4:
        perm_map_U4[d] = k
        
    U4 = U4_kron[perm_map_U4, :]
    
    # V4 has same structure (j,k) but wait, V is j*p+k.
    # In 2x2x2, V is j*2+k.
    # In 4x4x4, V is j*4+k.
    # Same permutation applies.
    V4 = V4_kron[perm_map_U4, :]
    
    # W4 has structure k*n+i.
    # In 2x2x2, W is k*2+i.
    # In 4x4x4, W is k*4+i.
    # Same permutation applies.
    W4 = W4_kron[perm_map_U4, :]
    
    # Step 3: Project to 3x4x4
    # We want i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    # U4 rows: i*4+j. Keep i<3. Rows 0..11.
    U344 = U4[0:12, :]
    
    # V4 rows: j*4+k. Keep all.
    V344 = V4
    
    # W4 rows: k*4+i. Keep i<3.
    # Rows:
    # k=0: 0,1,2,3 -> keep 0,1,2
    # k=1: 4,5,6,7 -> keep 4,5,6
    # k=2: 8,9,10,11 -> keep 8,9,10
    # k=3: 12,13,14,15 -> keep 12,13,14
    w_keep_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    W344 = W4[w_keep_indices, :]
    
    # Step 4: Add terms for k=4 part (3x4x1)
    # T[i,j,4] = A[i,j] * B[j,4]
    # This is a matrix M = A * diag(B[:,4])? No, A is 3x4, B[:,4] is 4x1.
    # M[i,j] = A[i,j] * B[j,4].
    # We need to decompose M into 3 rank-1 matrices.
    # Since rank(M) <= 3, we can use SVD or just pick 3 columns if A is full rank?
    # Actually, M = A * b where b = B[:,4].
    # M = sum_{j=0}^3 A[:,j] * (b[j] * e_j^T)? No.
    # M = sum_{j=0}^3 (A[:,j] * b[j]) * (e_j^T).
    # This is 4 terms. But we can do better.
    # Rank of M is at most 3.
    # We can find 3 vectors u_l, v_l such that M = sum u_l v_l^T.
    # Let's just use the first 3 columns of A scaled by b?
    # If we pick u_l = A[:,l] * b[l] and v_l = e_l, then sum is only for l=0,1,2.
    # Missing the j=3 term.
    # We need a proper rank-3 factorization.
    # Use SVD of M.
    # M is 3x4.
    # U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    # M = sum_{l=0}^{2} (U_svd[:,l] * S[l]) (Vt[l, :])^T
    # So u_l = U_svd[:,l] * S[l], v_l = Vt[l, :]
    
    # Since we don't have actual A and B values, we must provide the decomposition coefficients symbolically?
    # No, the task asks for a decomposition of the TENSOR.
    # The tensor T is defined by T[i,j,k] = 1 if k is the index of multiplication?
    # No, the tensor is the multiplication operator.
    # T[i, j, k] is 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i,j,k?
    # Yes, the standard matrix multiplication tensor has 1s at (i, j, k) corresponding to the product.
    # Wait, the tensor T_{n,m,p} has T[i, j, k] = 1 if the product A[i,j]B[j,k] is part of the sum for C[i,k].
    # In the standard definition, T[i, j, k] = 1 for all valid i,j,k.
    # So for the k=4 part, T[i, j, 4] = 1 for all i in 0..2, j in 0..3.
    # We need to decompose the sub-tensor T_sub[i, j] = 1 (for i in 0..2, j in 0..3) into 3 rank-1 tensors.
    # T_sub[i, j] = sum_{l=0}^2 u[i, l] v[j, l].
    # This is just decomposing a 3x4 matrix of 1s into 3 rank-1 matrices.
    # A 3x4 matrix of 1s has rank 1.
    # So we can do it in 1 term!
    # u = [1, 1, 1]^T, v = [1, 1, 1, 1]^T.
    # u v^T = Matrix of 1s.
    # So we only need 1 term for the k=4 part.
    
    # Let's verify.
    # Term: u_l (3x1), v_l (4x1), w_l (15x1).
    # w_l has 1 at index k=4, i=0..2?
    # W index is k*n + i.
    # For k=4, i=0: 12. i=1: 13. i=2: 14.
    # The term contributes to C[i, 4] via sum_j A[i,j] B[j,4] * (u[i] v[j] w[12+i])?
    # No, the decomposition is T = sum u_r \otimes v_r \otimes w_r.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For the k=4 part, we want T[i, j, 4] = 1.
    # So we need sum_r U[i*4+j, r] V[j*5+4, r] W[4*3+i, r] = 1.
    # If we use 1 term r=56:
    # U[i*4+j, 56] = 1 for all i,j.
    # V[j*5+4, 56] = 1 for all j.
    # W[4*3+i, 56] = 1 for all i.
    # Then the sum is 1*1*1 = 1. Correct.
    
    # So we add 1 term.
    # Total rank = 56 + 1 = 57.
    
    rank = 57
    U_final = np.zeros((12, rank)) # n*m = 3*4 = 12
    V_final = np.zeros((20, rank)) # m*p = 4*5 = 20
    W_final = np.zeros((15, rank)) # p*n = 5*3 = 15
    
    # Copy the 56 terms from 3x4x4 part
    # U344 is 12x56. Matches U_final first 56 cols.
    U_final[:, 0:56] = U344
    
    # V344 is 16x56. But V_final is 20x57.
    # V344 rows correspond to j*4+k for k in 0..3.
    # V_final rows correspond to j*5+k for k in 0..4.
    # We need to map V344 rows to V_final rows.
    # V344 row index: j*4+k.
    # V_final row index: j*5+k.
    # For k in 0..3, map j*4+k -> j*5+k.
    v_map = []
    for j in range(4):
        for k in range(4):
            v_map.append(j*4+k)
    # v_map is just 0..15.
    # We need to place these into V_final rows j*5+k.
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        v_final_idx = j * 5 + k
        V_final[v_final_idx, 0:56] = V344[idx, 0:56]
        
    # W344 is 12x56. Rows are mapped from W4.
    # W344 rows correspond to k*4+i for i<3.
    # W_final rows correspond to k*3+i.
    # We need to map W344 rows to W_final rows.
    # W344 row order: 
    # 0: k=0,i=0 -> W_final 0*3+0=0
    # 1: k=0,i=1 -> W_final 0*3+1=1
    # 2: k=0,i=2 -> W_final 0*3+2=2
    # 3: k=1,i=0 -> W_final 1*3+0=3
    # 4: k=1,i=1 -> W_final 1*3+1=4
    # 5: k=1,i=2 -> W_final 1*3+2=5
    # 6: k=2,i=0 -> W_final 2*3+0=6
    # 7: k=2,i=1 -> W_final 2*3+1=7
    # 8: k=2,i=2 -> W_final 2*3+2=8
    # 9: k=3,i=0 -> W_final 3*3+0=9
    # 10: k=3,i=1 -> W_final 3*3+1=10
    # 11: k=3,i=2 -> W_final 3*3+2=11
    
    w_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for idx in range(12):
        W_final[w_map[idx], 0:56] = W344[idx, 0:56]
        
    # Add the 1 term for k=4
    # U_final[:, 56] = 1
    U_final[:, 56] = 1
    # V_final[j*5+4, 56] = 1 for all j
    for j in range(4):
        V_final[j*5+4, 56] = 1
    # W_final[4*3+i, 56] = 1 for all i
    for i in range(3):
        W_final[4*3+i, 56] = 1
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
