"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 6x6x6 as 3x3 meta-matrix of 2x2 blocks.
    # Meta rank: 3*3*3 = 27 (schoolbook)
    # Block rank: 2*2*2 = 7 (Strassen)
    # Total rank: 27 * 7 = 189
    
    meta_n, meta_m, meta_p = 3, 3, 3
    block_n, block_m, block_p = 2, 2, 2
    
    rank = (meta_n * meta_m * meta_p) * (block_n * block_m * block_p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-compute Strassen factors for 2x2x2
    # Indices i,j,k in {0,1}
    # U_str[i*2+j, s], V_str[j*2+k, s], W_str[k*2+i, s]
    U_str = [[0] * 7 for _ in range(4)]
    V_str = [[0] * 7 for _ in range(4)]
    W_str = [[0] * 7 for _ in range(4)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to express the terms as rank-1 tensors.
    # Each M_s is a product of a linear form in A and a linear form in B.
    # The result C is a linear combination of M_s.
    
    # Let's define the coefficients for U, V, W for each of the 7 multiplications.
    # For each s, U_str[i,j,s] is the coeff of A_ij in the first factor.
    # V_str[j,k,s] is the coeff of B_jk in the second factor.
    # W_str[k,i,s] is the coeff of C_ki? No, W corresponds to the output.
    # The decomposition is sum_s U_str[i,j,s] V_str[j,k,s] W_str[k,i,s] = T_str[i,j,k].
    # Wait, the Strassen algorithm computes C from A and B.
    # It doesn't directly give a tensor decomposition of the form sum U V W = T.
    # However, the rank-7 decomposition of the 2x2x2 multiplication tensor is known.
    
    # The tensor T_str[i,j,k] is 1 if the term A_ij B_jk contributes to C_ik?
    # In 2x2, C_ik = sum_j A_ij B_jk.
    # The tensor is T[i,j,k].
    
    # Explicit Rank-7 Decomposition of 2x2x2 Multiplication Tensor:
    # Based on Strassen's identities.
    
    # We fill U_str, V_str, W_str such that:
    # T[i,j,k] = sum_s U_str[i,j,s] * V_str[j,k,s] * W_str[k,i,s]
    
    # Note: The indices in W are k, i.
    
    # s=0: M1
    # A part: A11 + A22 -> U[1,1]=1, U[2,2]=1
    # B part: B11 + B22 -> V[1,1]=1, V[2,2]=1
    # C part: C11 + C22? No, M1 contributes to C11 and C22.
    # C11 = M1 + ...
    # C22 = M1 + ...
    # So W[1,1] (C11) gets +1? W index is k*n+i.
    # For C11, i=0, k=0? Or i=1, k=1?
    # Let's use 0-based indices.
    # C[0,0] corresponds to i=0, k=0?
    # W index is k*2 + i.
    # C[0,0] -> W[0*2+0] = W[0].
    # C[1,1] -> W[1*2+1] = W[3].
    
    # M1 contributes to C[0,0] and C[1,1].
    # So W[0] += 1, W[3] += 1.
    
    U_str[0*2+0][0] = 1; U_str[1*2+1][0] = 1
    V_str[0*2+0][0] = 1; V_str[1*2+1][0] = 1
    W_str[0*2+0][0] = 1; W_str[1*2+1][0] = 1
    
    # s=1: M2 = (A21+A22)B11
    # A: A[1,0] + A[1,1] -> U[1,0]=1, U[1,1]=1
    # B: B[0,0] -> V[0,0]=1
    # C: M2 contributes to C[1,0] (C21).
    # C[1,0] -> i=1, k=0? No, C[i,k].
    # C[1,0] means row 1, col 0.
    # W index k*2+i = 0*2+1 = 1.
    W_str[0*2+1][1] = 1
    U_str[1*2+0][1] = 1; U_str[1*2+1][1] = 1
    V_str[0*2+0][1] = 1
    
    # s=2: M3 = A11(B12-B22)
    # A: A[0,0] -> U[0,0]=1
    # B: B[0,1] - B[1,1] -> V[0,1]=1, V[1,1]=-1
    # C: M3 contributes to C[0,1] (C12).
    # C[0,1] -> i=0, k=1. W index 1*2+0 = 2.
    W_str[1*2+0][2] = 1
    U_str[0*2+0][2] = 1
    V_str[0*2+1][2] = 1; V_str[1*2+1][2] = -1
    
    # s=3: M4 = A22(B21-B11)
    # A: A[1,1] -> U[1,1]=1
    # B: B[1,0] - B[0,0] -> V[1,0]=1, V[0,0]=-1
    # C: M4 contributes to C[1,1] (C22) and C[0,0] (C11).
    # C[1,1] -> i=1, k=1. W index 1*2+1 = 3.
    # C[0,0] -> i=0, k=0. W index 0*2+0 = 0.
    W_str[1*2+1][3] = 1; W_str[0*2+0][3] = 1
    U_str[1*2+1][3] = 1
    V_str[1*2+0][3] = 1; V_str[0*2+0][3] = -1
    
    # s=4: M5 = (A11+A12)B22
    # A: A[0,0] + A[0,1] -> U[0,0]=1, U[0,1]=1
    # B: B[1,1] -> V[1,1]=1
    # C: M5 contributes to C[0,1] (C12) and C[1,1] (C22).
    # C[0,1] -> W[2]. C[1,1] -> W[3].
    W_str[1*2+0][4] = 1; W_str[1*2+1][4] = 1
    U_str[0*2+0][4] = 1; U_str[0*2+1][4] = 1
    V_str[1*2+1][4] = 1
    
    # s=5: M6 = (A21-A11)(B11+B12)
    # A: A[1,0] - A[0,0] -> U[1,0]=1, U[0,0]=-1
    # B: B[0,0] + B[0,1] -> V[0,0]=1, V[0,1]=1
    # C: M6 contributes to C[1,1] (C22).
    # C[1,1] -> W[3].
    W_str[1*2+1][5] = 1
    U_str[1*2+0][5] = 1; U_str[0*2+0][5] = -1
    V_str[0*2+0][5] = 1; V_str[0*2+1][5] = 1
    
    # s=6: M7 = (A12-A22)(B21+B22)
    # A: A[0,1] - A[1,1] -> U[0,1]=1, U[1,1]=-1
    # B: B[1,0] + B[1,1] -> V[1,0]=1, V[1,1]=1
    # C: M7 contributes to C[0,0] (C11).
    # C[0,0] -> W[0].
    W_str[0*2+0][6] = 1
    U_str[0*2+1][6] = 1; U_str[1*2+1][6] = -1
    V_str[1*2+0][6] = 1; V_str[1*2+1][6] = 1
    
    # Now build the full 6x6x6 decomposition
    # Meta indices I, J, K in 0..2
    # Block indices i, j, k in 0..1
    # Full indices:
    # U_row = (2I+i)*6 + (2J+j)
    # V_col = (2J+j)*6 + (2K+k)
    # W_idx = (2K+k)*6 + (2I+i)
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                meta_r = I * 3 + J * 3 + K # Arbitrary unique ID for meta term? No, meta_r is just an index 0..26.
                # Actually, we just need to iterate.
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            # This combination (I,J,K,i,j,k) corresponds to a specific term in the schoolbook expansion of the meta-matrix.
                            # But we are using Strassen for the block.
                            # The meta-term is just 1 (schoolbook).
                            # So for each (I,J,K), we add 7 terms (one for each Strassen s).
                            
                            # Wait, the meta decomposition is schoolbook.
                            # Meta term r corresponds to (I,J,K).
                            # U_meta[I,J] = 1, V_meta[J,K] = 1, W_meta[K,I] = 1.
                            
                            for s in range(7):
                                col = (I * 3 + J * 3 + K) * 7 + s # This mapping is not quite right for U/V/W alignment.
                                # We need to align the meta index with the block index.
                                # The meta index is just a label.
                                # Let's use a global column index.
                                
                                # We will fill U, V, W directly.
                                pass

    # Let's restart the filling loop properly.
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Meta term is 1 for this (I,J,K)
                # For each Strassen component s
                for s in range(7):
                    # U part
                    for i in range(2):
                        for j in range(2):
                            u_val = U_str[i*2+j][s]
                            if u_val != 0:
                                # U index: (2I+i)*6 + (2J+j)
                                u_idx = (2*I + i) * 6 + (2*J + j)
                                U[u_idx][col] = u_val
                    # V part
                    for j in range(2):
                        for k in range(2):
                            v_val = V_str[j*2+k][s]
                            if v_val != 0:
                                # V index: (2J+j)*6 + (2K+k)
                                v_idx = (2*J + j) * 6 + (2*K + k)
                                V[v_idx][col] = v_val
                    # W part
                    for k in range(2):
                        for i in range(2):
                            w_val = W_str[k*2+i][s]
                            if w_val != 0:
                                # W index: (2K+k)*6 + (2I+i)
                                w_idx = (2*K + k) * 6 + (2*I + i)
                                W[w_idx][col] = w_val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
