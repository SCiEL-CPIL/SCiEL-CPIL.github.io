import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition using the Kronecker product of the 2x2 Strassen tensor
    # and the 2x2 Naive (identity) tensor, combined with handling for the "5th" dimension.
    
    # 1. Define the 2x2 Strassen decomposition (Rank 7)
    # U, V, W are 4x7 matrices.
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, -1, 0, 0, -1]
    ], dtype=float)
    
    S_V = np.array([
        [1, 0, 0, 1, 0, -1, 0],
        [0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, -1]
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # 2. Define the 2x2 Naive decomposition (Rank 4)
    # This is the standard element-wise multiplication tensor for 2x2 matrices.
    # U, V, W are 4x4 matrices.
    N_U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    N_V = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    # Wait, let's verify N_V and N_W for naive 2x2.
    # Terms:
    # 0: A00 B00 -> C00. U=e0, V=e0, W=e0.
    # 1: A01 B10 -> C01. U=e1, V=e2, W=e1.
    # 2: A10 B00 -> C10. U=e2, V=e0, W=e2.
    # 3: A11 B10 -> C11. U=e3, V=e2, W=e3.
    # 4: A00 B01 -> C01. U=e0, V=e1, W=e1.
    # 5: A01 B11 -> C02? No C01 is index 1. C11 is index 3.
    # Let's map indices: C00=0, C01=1, C10=2, C11=3.
    # Term 0: A00 B00. U=e0, V=e0, W=e0.
    # Term 1: A01 B10. U=e1, V=e2, W=e1.
    # Term 2: A10 B00. U=e2, V=e0, W=e2.
    # Term 3: A11 B10. U=e3, V=e2, W=e3.
    # Term 4: A00 B01. U=e0, V=e1, W=e1.
    # Term 5: A01 B11. U=e1, V=e3, W=e3.
    # Term 6: A10 B01. U=e2, V=e1, W=e2.
    # Term 7: A11 B11. U=e3, V=e3, W=e3.
    
    N_U = np.array([
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ], dtype=float)
    
    N_V = np.array([
        [1, 0, 1, 0], # B00 for terms 0,2? No.
        # Term 0: V=e0 (B00)
        # Term 1: V=e2 (B10)
        # Term 2: V=e0 (B00)
        # Term 3: V=e2 (B10)
        # Term 4: V=e1 (B01)
        # Term 5: V=e3 (B11)
        # Term 6: V=e1 (B01)
        # Term 7: V=e3 (B11)
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    
    N_W = np.array([
        [1, 0, 0, 0], # C00
        [0, 1, 0, 1], # C01 (Terms 1, 4)
        [0, 0, 1, 0], # C10 (Terms 2, 6)
        [0, 0, 0, 1]  # C11 (Terms 3, 5, 7) -> Wait.
        # Term 3: W=e3 (C11)
        # Term 5: W=e3 (C11)
        # Term 7: W=e3 (C11)
        # So W row 3 has 1s at cols 3,5,7.
        # My N_W above is wrong.
        
        # Let's rebuild N_W carefully.
        # Col 0: Term 0 -> W=e0.
        # Col 1: Term 1 -> W=e1.
        # Col 2: Term 2 -> W=e2.
        # Col 3: Term 3 -> W=e3.
        # Col 4: Term 4 -> W=e1.
        # Col 5: Term 5 -> W=e3.
        # Col 6: Term 6 -> W=e2.
        # Col 7: Term 7 -> W=e3.
        
        # N_W (4x8)
        [1, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 1, 0, 0, 1, 0, 0, 0], # C01
        [0, 0, 1, 0, 0, 0, 1, 0], # C10
        [0, 0, 0, 1, 0, 1, 0, 1]  # C11
    ], dtype=float)
    
    N_U = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1]
    ], dtype=float)
    
    N_V = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype=float)
    
    # 3. Construct 4x4x4 Decomposition
    # We want to compute C4 = A4 B4.
    # We can view A4 as a 2x2 block matrix of 2x2 blocks.
    # A4 = [[A00, A01], [A10, A11]]
    # B4 = [[B00, B01], [B10, B11]]
    # C4 = [[C00, C01], [C10, C11]]
    
    # The Kronecker product of Strassen (2x2) and Naive (2x2) gives a decomposition for 4x4.
    # Rank = 7 * 4 = 28.
    
    # The indices for the 4x4 case:
    # A index i in 0..3. i = 2*i_s + i_n.
    # B index j in 0..3. j = 2*j_s + j_n.
    # C index k in 0..3. k = 2*k_s + k_n.
    
    # The Kronecker tensor T_{i,j,k} = sum_{c_s, c_n} S_U[i_s, c_s] S_V[j_s, c_s] S_W[k_s, c_s] N_U[i_n, c_n] N_V[j_n, c_n] N_W[k_n, c_n]
    
    # We need to map this to the 4x5x5 problem.
    # A is 4x5. B is 5x5. C is 4x5.
    # We split the 5 dimensions into 4 (handled by Kronecker) and 1 (handled naively).
    # A = [A_4, A_1] where A_4 is 4x4, A_1 is 4x1.
    # B = [B_4, B_1] where B_4 is 4x4, B_1 is 4x1? No, B is 5x5.
    # B = [[B_44, B_41], [B_14, B_11]]
    # C = [C_4, C_1] where C_4 is 4x4, C_1 is 4x1.
    
    # C_4 = A_4 B_44 + A_1 B_14
    # C_1 = A_4 B_41 + A_1 B_11
    
    # We use the rank-28 decomposition for A_4 B_44.
    # We use naive decompositions for the other terms.
    
    # Term 1: A_4 B_44 (Rank 28)
    # Term 2: A_4 B_41 (4x4 * 4x1). Rank 16.
    # Term 3: A_1 B_14 (4x1 * 1x4). Rank 4.
    # Term 4: A_1 B_11 (4x1 * 1x1). Rank 4.
    
    # Total Rank = 28 + 16 + 4 + 4 = 52.
    # This is significantly better than 85.
    
    # Let's verify the indices and construct the matrices.
    
    total_rank = 52
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # 1. A_4 B_44 (Rank 28)
    # i in 0..3, j in 0..3, k in 0..3.
    # A index: i*m + j = i*5 + j. (j is 0..3)
    # B index: j*p + k = j*5 + k. (j is 0..3, k is 0..3)
    # C index: k*n + i = k*4 + i. (k is 0..3, i is 0..3)
    
    for c_s in range(7):
        for c_n in range(4):
            # This term corresponds to S_U[i_s, c_s] S_V[j_s, c_s] S_W[k_s, c_s] N_U[i_n, c_n] N_V[j_n, c_n] N_W[k_n, c_n]
            # We need to iterate over i_s, j_s, k_s, i_n, j_n, k_n.
            # But we can just compute the values for each i, j, k.
            
            for i in range(4):
                i_s, i_n = divmod(i, 2)
                u_val = S_U[i_s, c_s] * N_U[i_n, c_n]
                if u_val != 0:
                    U[i * 5 + (i_s * 2 + i_n), col] += u_val # Wait, j is the second index of A.
                    # A index is i*m + j. Here j is the column of A, which is the row of B.
                    # In the 4x4 block, j is 0..3.
                    # So A index is i*5 + j.
                    # My loop variable `i` is the row of A. `j` is the column of A (row of B).
                    # Let's rename loop variables to avoid confusion.
                    pass
            
            # Let's restart the loop structure for clarity.
            # For each Kronecker term (c_s, c_n):
            # U[i, col] = S_U[i_s, c_s] * N_U[i_n, c_n] where i = 2*i_s + i_n.
            # V[j, col] = S_V[j_s, c_s] * N_V[j_n, c_n] where j = 2*j_s + j_n.
            # W[k, col] = S_W[k_s, c_s] * N_W[k_n, c_n] where k = 2*k_s + k_n.
            
            # But we must map these to the global indices.
            # U global index: row = i, col_in_A = j.
            # Wait, U is (n*m, rank). Row index is i*m + j.
            # In the 4x4 block, i is 0..3, j is 0..3.
            # So U[i*5 + j, col] = S_U[i_s, c_s] * N_U[i_n, c_n] ?
            # No, the Kronecker product acts on the indices.
            # The tensor is T_{i,j,k}.
            # U contributes to the A part. A_{i,j}.
            # In the 4x4 block, A_{i,j} has i in 0..3, j in 0..3.
            # The Kronecker factorization of the 4x4 tensor T4_{i,j,k} is:
            # T4_{i,j,k} = sum_{c_s, c_n} U4_{i, c_s, c_n} V4_{j, c_s, c_n} W4_{k, c_s, c_n}
            # Where U4_{i, c_s, c_n} = S_U[i_s, c_s] N_U[i_n, c_n]?
            # No, the Kronecker product of tensors T1 and T2 is T1_{i1,j1,k1} T2_{i2,j2,k2}.
            # The factorization is (U1_{i1, c1} U2_{i2, c2}) (V1_{j1, c1} V2_{j2, c2}) (W1_{k1, c1} W2_{k2, c2}).
            # So U4_{i, (c1,c2)} = U1_{i1, c1} U2_{i2, c2}.
            # Here i = i1 * 2 + i2? No, i1 is the block index, i2 is the intra-block index.
            # Usually i = i1 * size2 + i2.
            # So i_s is i1, i_n is i2.
            # U4_{i, c} = S_U[i_s, c_s] * N_U[i_n, c_n].
            
            # So for the 4x4 block:
            # U4_{i, c} is non-zero for i in 0..3.
            # V4_{j, c} is non-zero for j in 0..3.
            # W4_{k, c} is non-zero for k in 0..3.
            
            # Mapping to global indices:
            # U global: row = i*5 + j.
            # Wait, U factor multiplies A_{i,j}.
            # So U[i*5 + j, col] = U4_{i, c} * V4_{j, c}? No.
            # The decomposition is sum_c U_{i,j, c} V_{j,k, c} W_{k,i, c}.
            # U_{i,j, c} depends on i and j?
            # No, in the standard tensor decomposition T_{i,j,k} = sum_c u_{i,c} v_{j,c} w_{k,c}.
            # Here i is row of A, j is col of A (row of B), k is col of B (col of C).
            # So u_{i,c} is associated with A row i? No.
            # The tensor is T_{i,j,k} = 1 if A_{i,j} B_{j,k} -> C_{i,k}.
            # The factors are:
            # U: size n*m x rank. Index (i, j).
            # V: size m*p x rank. Index (j, k).
            # W: size p*n x rank. Index (k, i).
            
            # In the Kronecker product for 4x4:
            # T4_{i,j,k} = sum_c U4_{(i,j), c} V4_{(j,k), c} W4_{(k,i), c}.
            # U4_{(i,j), c} = S_U[i_s, c_s] S_V[j_s, c_s] ? No.
            # The Strassen decomposition is for the 2x2 tensor T2_{I,J,K}.
            # T2_{I,J,K} = sum_{c_s} S_U[I, c_s] S_V[J, c_s] S_W[K, c_s].
            # The Naive decomposition is for the 2x2 tensor Tn_{i,j,k}.
            # Tn_{i,j,k} = sum_{c_n} N_U[i, c_n] N_V[j, c_n] N_W[k, c_n].
            
            # The 4x4 tensor T4_{i,j,k} where i = 2*I + i, j = 2*J + j, k = 2*K + k.
            # T4_{i,j,k} = T2_{I,J,K} * Tn_{i,j,k}.
            # So T4_{i,j,k} = sum_{c_s, c_n} (S_U[I, c_s] N_U[i, c_n]) (S_V[J, c_s] N_V[j, c_n]) (S_W[K, c_s] N_W[k, c_n]).
            
            # So the factors for the 4x4 case are:
            # U4_{(I,i, J,j), c} = S_U[I, c_s] N_U[i, c_n].
            # V4_{(J,j, K,k), c} = S_V[J, c_s] N_V[j, c_n].
            # W4_{(K,k, I,i), c} = S_W[K, c_s] N_W[k, c_n].
            
            # Now map to global 4x5x5.
            # A is 4x5. Rows 0..3. Cols 0..4.
            # B is 5x5. Rows 0..4. Cols 0..4.
            # C is 4x5. Rows 0..3. Cols 0..4.
            
            # The 4x4 block uses A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4].
            # For U4:
            # Global U index: (i, j) where i in 0..3, j in 0..3.
            # U[i*5 + j, col] = S_U[i_s, c_s] N_U[i_n, c_n].
            # Note: i = 2*i_s + i_n. j = 2*j_s + j_n.
            # But U only depends on i? No, U is for A_{i,j}.
            # In the tensor T_{i,j,k}, the U factor is associated with the (i,j) index of A.
            # So U4_{(i,j), c} = S_U[i_s, c_s] * S_V[j_s, c_s] * N_U[i_n, c_n] * N_V[j_n, c_n]?
            # No, the Kronecker product of two tensors T1 and T2 gives a tensor T1_{i1,j1,k1} T2_{i2,j2,k2}.
            # The factorization is (u1_{i1,c1} u2_{i2,c2}) (v1_{j1,c1} v2_{j2,c2}) (w1_{k1,c1} w2_{k2,c2}).
            # So the U factor for the combined tensor is u1_{i1,c1} u2_{i2,c2}.
            # Here i1 is the block index I, i2 is the intra-block index i.
            # So U_combined_{(I,i), c} = S_U[I, c_s] * N_U[i, c_n].
            # This U factor is associated with the row index of the combined matrix.
            # The combined matrix A4 has rows indexed by (I,i) which is i_global = 2*I + i.
            # But A4 is a 4x4 matrix. The "row" of A4 is i_global.
            # However, the U factor in the decomposition of T4_{i,j,k} is U_{i,j, c}.
            # Wait, the standard decomposition is T_{i,j,k} = sum_c u_{i,c} v_{j,c} w_{k,c}.
            # Here u_{i,c} is a vector of length n*m? No.
            # The problem statement says:
            # U: (n*m, rank). V: (m*p, rank). W: (p*n, rank).
            # This implies U_{i*m+j, c} is the factor for A_{i,j}.
            # So U_{i*m+j, c} = u_{i,c} * v_{j,c}? No, that would be a product of two vectors.
            # In the Kronecker product case:
            # U4_{i_global * 4 + j_global, c} = S_U[I, c_s] N_U[i, c_n] * S_V[J, c_s] N_V[j, c_n]?
            # No, the U factor in the tensor decomposition T_{i,j,k} = sum u_{i,j,c} v_{j,k,c} w_{k,i,c}
            # is u_{i,j,c}.
            # In the Kronecker product T4 = T2 \otimes Tn:
            # T4_{i,j,k} = T2_{I,J,K} Tn_{i,j,k}.
            # Factorization:
            # u4_{i,j,c} = u2_{I,J,c_s} * un_{i,j,c_n}.
            # v4_{j,k,c} = v2_{J,K,c_s} * vn_{j,k,c_n}.
            # w4_{k,i,c} = w2_{K,I,c_s} * wn_{k,i,c_n}.
            
            # So we need u2_{I,J,c_s}.
            # For Strassen, the decomposition is T2_{I,J,K} = sum_c S_U[I,c] S_V[J,c] S_W[K,c].
            # So u2_{I,J,c} = S_U[I,c] * S_V[J,c].
            # Similarly for Naive: u_n_{i,j,c} = N_U[i,c] * N_V[j,c].
            
            # So U4_{i_global, j_global, c} = (S_U[I,c_s] S_V[J,c_s]) * (N_U[i,c_n] N_V[j,c_n]).
            # Where i_global = 2*I + i, j_global = 2*J + j.
            
            # Let's implement this.
            
            u_val = S_U[i_s, c_s] * S_V[j_s, c_s] * N_U[i_n, c_n] * N_V[j_n, c_n]
            # Wait, i_s, i_n come from i_global. j_s, j_n from j_global.
            # We need to iterate over i_global, j_global.
            
            pass
            
    # Let's simplify. We will build the U, V, W matrices for the 4x4 block first, then map to global.
    
    # 4x4 Block Decomposition (Rank 28)
    U4 = np.zeros((4 * 4, 28))
    V4 = np.zeros((4 * 4, 28))
    W4 = np.zeros((4 * 4, 28))
    
    for c_s in range(7):
        for c_n in range(4):
            c = c_s * 4 + c_n
            for I in range(2):
                for i in range(2):
                    i_global = 2 * I + i
                    for J in range(2):
                        for j in range(2):
                            j_global = 2 * J + j
                            u_val = S_U[I, c_s] * S_V[J, c_s] * N_U[i, c_n] * N_V[j, c_n]
                            if u_val != 0:
                                U4[i_global * 4 + j_global, c] = u_val
                            
                            v_val = S_V[J, c_s] * S_W[K, c_s] * N_V[j, c_n] * N_W[k, c_n] # Need K,k
                            # This is getting messy. Let's just compute the full tensor and factorize?
                            # No, we want the factors.
                            
                            # Let's do it for U, V, W separately.
                            # U4[i_global * 4 + j_global, c] = S_U[I, c_s] S_V[J, c_s] N_U[i, c_n] N_V[j, c_n]
                            # V4[j_global * 4 + k_global, c] = S_V[J, c_s] S_W[K, c_s] N_V[j, c_n] N_W[k, c_n]
                            # W4[k_global * 4 + i_global, c] = S_W[K, c_s] S_U[I, c_s] N_W[k, c_n] N_U[i, c_n]
                            
                            # We need to iterate over K,k for V and W.
                            pass
    
    # Let's rewrite the loop properly.
    
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    for c_s in range(7):
        for c_n in range(4):
            c = c_s * 4 + c_n
            for I in range(2):
                for i in range(2):
                    i_g = 2 * I + i
                    for J in range(2):
                        for j in range(2):
                            j_g = 2 * J + j
                            u_val = S_U[I, c_s] * S_V[J, c_s] * N_U[i, c_n] * N_V[j, c_n]
                            if u_val != 0:
                                U4[i_g * 4 + j_g, c] = u_val
                            
                            for K in range(2):
                                for k in range(2):
                                    k_g = 2 * K + k
                                    v_val = S_V[J, c_s] * S_W[K, c_s] * N_V[j, c_n] * N_W[k, c_n]
                                    if v_val != 0:
                                        V4[j_g * 4 + k_g, c] = v_val
                                    
                                    w_val = S_W[K, c_s] * S_U[I, c_s] * N_W[k, c_n] * N_U[i, c_n]
                                    if w_val != 0:
                                        W4[k_g * 4 + i_g, c] = w_val
                                

    # Map U4, V4, W4 to global U, V, W for the 4x4 block part.
    # A is 4x5. The 4x4 block is A[0:4, 0:4].
    # U4 row index: i_g * 4 + j_g.
    # Global U row index: i_g * 5 + j_g. (j_g is 0..3)
    # So U[i_g * 5 + j_g, col] = U4[i_g * 4 + j_g, c].
    
    for i_g in range(4):
        for j_g in range(4):
            for c in range(28):
                U[i_g * 5 + j_g, c] = U4[i_g * 4 + j_g, c]
                
    # V4 row index: j_g * 4 + k_g.
    # Global V row index: j_g * 5 + k_g. (j_g, k_g are 0..3)
    # So V[j_g * 5 + k_g, col] = V4[j_g * 4 + k_g, c].
    
    for j_g in range(4):
        for k_g in range(4):
            for c in range(28):
                V[j_g * 5 + k_g, c] = V4[j_g * 4 + k_g, c]
                
    # W4 row index: k_g * 4 + i_g.
    # Global W row index: k_g * 4 + i_g. (k_g, i_g are 0..3)
    # So W[k_g * 4 + i_g, col] = W4[k_g * 4 + i_g, c].
    
    for k_g in range(4):
        for i_g in range(4):
            for c in range(28):
                W[k_g * 4 + i_g, c] = W4[k_g * 4 + i_g, c]

    col = 28
    
    # 2. A_4 B_41 (4x4 * 4x1). Rank 16.
    # A_4 is 4x4. B_41 is 4x1 (col 4 of B).
    # C_1 is 4x1 (col 4 of C).
    # C_1[i] = sum_j A_4[i,j] B_41[j].
    # Naive: 16 terms.
    # Term (i,j): A_{i,j} B_{j,4} -> C_{i,4}.
    # U[i*5 + j, col] = 1.
    # V[j*5 + 4, col] = 1.
    # W[4*4 + i, col] = 1.
    
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    # 3. A_1 B_14 (4x1 * 1x4). Rank 4.
    # A_1 is 4x1 (col 4 of A). B_14 is 1x4 (row 4 of B).
    # C_4 is 4x4.
    # C_4[i,k] = A_1[i] B_14[k].
    # Naive: 4 terms? No, 4*4 = 16?
    # C_4[i,k] = A_{i,4} B_{4,k}.
    # This is an outer product.
    # We can do it with 4 terms?
    # C_4 = A_1 B_14.
    # This is rank 1 tensor?
    # T_{i,4,k} = A_{i,4} B_{4,k}.
    # We can write this as sum_c u_{i,4,c} v_{4,k,c} w_{k,i,c}.
    # If we use 1 term: u_{i,4,0} = A_{i,4}, v_{4,k,0} = B_{4,k}, w_{k,i,0} = 1?
    # No, w must be a factor.
    # The tensor is T_{i,j,k} where j=4.
    # T_{i,4,k} = A_{i,4} B_{4,k}.
    # We can use 1 term:
    # U[i*5 + 4, col] = 1.
    # V[4*5 + k, col] = 1.
    # W[k*4 + i, col] = 1.
    # Then sum_c U V W = sum_k A_{i,4} B_{4,k} = C_{i,k}.
    # So Rank 1?
    # Let's check.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # For j=4: C_{i,k} += A_{i,4} B_{4,k}.
    # Our term: U_{i,4} V_{4,k} W_{k,i}.
    # If U_{i,4}=1, V_{4,k}=1, W_{k,i}=1, then sum is 1*1*1 = 1.
    # But we need A_{i,4} B_{4,k}.
    # The factors multiply the inputs.
    # The decomposition is exact if T = sum u v w.
    # T_{i,4,k} = 1 (if we consider the structure).
    # No, the tensor T is defined by T_{i,j,k} = 1 if i,j,k match the multiplication structure.
    # For the term A_{i,4} B_{4,k} -> C_{i,k}, the tensor entry is 1 at (i,4,k).
    # So we need a rank-1 decomposition of the tensor E_{i,4,k}.
    # E_{i,4,k} = 1 if j=4.
    # We can use U_{i,4}=1, V_{4,k}=1, W_{k,i}=1?
    # Then sum_{i,j,k} U_{i,j} V_{j,k} W_{k,i} A_{i,j} B_{j,k} = sum_{i,k} 1 * 1 * 1 * A_{i,4} B_{4,k} = sum_{i,k} A_{i,4} B_{4,k}.
    # This contributes to C_{i,k} correctly.
    # So Rank 1 is sufficient for A_1 B_14.
    
    U[:, col] = 0
    V[:, col] = 0
    W[:, col] = 0
    for i in range(4):
        U[i * 5 + 4, col] = 1
    for k in range(4):
        V[4 * 5 + k, col] = 1
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, col] = 1
    col += 1
            
    # 4. A_1 B_11 (4x1 * 1x1). Rank 1.
    # C_1[i] = A_{i,4} B_{4,4}.
    # Term: U_{i,4}=1, V_{4,4}=1, W_{4,i}=1.
    
    U[:, col] = 0
    V[:, col] = 0
    W[:, col] = 0
    for i in range(4):
        U[i * 5 + 4, col] = 1
    V[4 * 5 + 4, col] = 1
    for i in range(4):
        W[4 * 4 + i, col] = 1
    col += 1
    
    # Total rank used: 28 + 16 + 1 + 1 = 46.
    
    # Wait, did I handle A_4 B_41 correctly?
    # C_1[i] = sum_j A_{i,j} B_{j,4}.
    # I used 16 terms, one for each (i,j).
    # Can I do better?
    # T_{i,j,4} = 1.
    # This is the identity tensor I_4 for the j index?
    # C_1 = A_4 B_41.
    # This is matrix-vector multiplication.
    # Rank of identity tensor I_n is n^2.
    # So 16 is optimal for this part.
    
    # Can I combine A_1 B_11 with A_1 B_14?
    # A_1 B_11: C_1[i] = A_{i,4} B_{4,4}.
    # A_1 B_14: C_4[i,k] = A_{i,4} B_{4,k}.
    # They involve different C indices (C_1 vs C_4).
    # So they cannot be combined easily.
    
    # Total Rank = 46.
    
    # Let's double check the Strassen factors.
    # S_U, S_V, S_W are correct?
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # My S_U:
    # Row 0 (A11): 1, 0, 1, 0, 1, -1, 0
    # Row 1 (A12): 0, 0, 0, 1, 1, 0, 1
    # Row 2 (A21): 1, 1, 0, 0, 0, 1, 0
    # Row 3 (A22): 0, 1, 1, -1, 0, 0, -1
    
    # M1: A11+A22. U: 1,0. V: 1,0.
    # M2: A21+A22. U: 1,1. V: 1,0.
    # M3: A11. U: 1,0. V: 0,1.
    # M4: A22. U: 0,1. V: 1,0.
    # M5: A11+A12. U: 1,1. V: 0,1.
    # M6: A21-A11. U: 1,-1. V: 1,1.
    # M7: A12-A22. U: 0,1. V: 1,1.
    
    # My S_U columns:
    # 0: 1,0,1,0 (M1: A11+A22? No, A11=1, A22=0. Should be 1,1? No, M1 is (A11+A22). So A11 coeff 1, A22 coeff 1.
    # My col 0: 1,0,1,0. This is A11+A21.
    # This is not Strassen.
    
    # Let's use a known correct Strassen decomposition.
    # From Wikipedia:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U (A factors):
    # M1: A11=1, A22=1.
    # M2: A21=1, A22=1.
    # M3: A11=1.
    # M4: A22=1.
    # M5: A11=1, A12=1.
    # M6: A21=1, A11=-1.
    # M7: A12=1, A22=-1.
    
    # V (B factors):
    # M1: B11=1, B22=1.
    # M2: B11=1.
    # M3: B21=1, B22=-1.
    # M4: B12=1, B11=-1.
    # M5: B22=1.
    # M6: B11=1, B12=1.
    # M7: B21=1, B22=1.
    
    # W (C factors):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # S_U (4x7):
    # Rows: A11, A12, A21, A22
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: -1, 0, 1, 0
    # M7: 0, 1, 0, -1
    
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1]
    ], dtype=float)
    # Check M1: B11=1, B22=1. Row 0: 1,0. Row 3: 1,0. Correct.
    # Check M2: B11=1. Row 0: 1. Correct.
    # Check M3: B21=1, B22=-1. Row 2: 1. Row 3: -1. Correct.
    # Check M4: B12=1, B11=-1. Row 1: 1. Row 0: -1. Correct.
    # Check M5: B22=1. Row 3: 1. Correct.
    # Check M6: B11=1, B12=1. Row 0: 1. Row 1: 1. Correct.
    # Check M7: B21=1, B22=1. Row 2: 1. Row 3: 1. Correct.
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, 0]
    ], dtype=float)
    # Check C11 (Row 0): M1=1, M4=1, M5=-1, M7=1.
    # My Row 0: 1, 0, 0, 1, -1, 0, 1. Correct.
    # Check C12 (Row 1): M3=1, M5=1.
    # My Row 1: 0, 0, 1, 0, 1, 0, 0. Correct.
    # Check C21 (Row 2): M2=1, M4=1.
    # My Row 2: 0, 1, 0, 1, 0, 0, 0. Correct.
    # Check C22 (Row 3): M1=1, M2=-1, M3=0, M6=1.
    # My Row 3: 1, -1, 0, 0, 0, 1, 0. Correct.
    
    # Re-run the construction with correct Strassen factors.
    
    total_rank = 46
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # 1. 4x4 Block (Rank 28)
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    for c_s in range(7):
        for c_n in range(4):
            c = c_s * 4 + c_n
            for I in range(2):
                for i in range(2):
                    i_g = 2 * I + i
                    for J in range(2):
                        for j in range(2):
                            j_g = 2 * J + j
                            u_val = S_U[I, c_s] * S_V[J, c_s] * N_U[i, c_n] * N_V[j, c_n]
                            if u_val != 0:
                                U4[i_g * 4 + j_g, c] = u_val
                            
                            for K in range(2):
                                for k in range(2):
                                    k_g = 2 * K + k
                                    v_val = S_V[J, c_s] * S_W[K, c_s] * N_V[j, c_n] * N_W[k, c_n]
                                    if v_val != 0:
                                        V4[j_g * 4 + k_g, c] = v_val
                                    
                                    w_val = S_W[K, c_s] * S_U[I, c_s] * N_W[k, c_n] * N_U[i, c_n]
                                    if w_val != 0:
                                        W4[k_g * 4 + i_g, c] = w_val
                                

    for i_g in range(4):
        for j_g in range(4):
            for c in range(28):
                U[i_g * 5 + j_g, c] = U4[i_g * 4 + j_g, c]
                
    for j_g in range(4):
        for k_g in range(4):
            for c in range(28):
                V[j_g * 5 + k_g, c] = V4[j_g * 4 + k_g, c]
                
    for k_g in range(4):
        for i_g in range(4):
            for c in range(28):
                W[k_g * 4 + i_g, c] = W4[k_g * 4 + i_g, c]

    col = 28
    
    # 2. A_4 B_41 (Rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    # 3. A_1 B_14 (Rank 1)
    U[:, col] = 0
    V[:, col] = 0
    W[:, col] = 0
    for i in range(4):
        U[i * 5 + 4, col] = 1
    for k in range(4):
        V[4 * 5 + k, col] = 1
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, col] = 1
    col += 1
            
    # 4. A_1 B_11 (Rank 1)
    U[:, col] = 0
    V[:, col] = 0
    W[:, col] = 0
    for i in range(4):
        U[i * 5 + 4, col] = 1
    V[4 * 5 + 4, col] = 1
    for i in range(4):
        W[4 * 4 + i, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
