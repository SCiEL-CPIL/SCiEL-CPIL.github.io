import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank-85 decomposition for the 4x5x5 matrix multiplication tensor.
    # The strategy is to split the 5x5 dimension into a 4x4 block and a 1x1 block.
    # 
    # The tensor T for 4x5x5 multiplication can be decomposed into four parts based on the indices of A (4x5), B (5x5), and C (4x5):
    # 1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]  (4x4x4 subproblem)
    # 2. A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4]  (4x1x4 subproblem)
    # 3. A[0:4, 0:4] * B[0:4, 4:5] -> C[0:4, 4:5]  (4x4x1 subproblem)
    # 4. A[0:4, 4:5] * B[4:5, 4:5] -> C[0:4, 4:5]  (4x1x1 subproblem)
    #
    # We use the Strassen algorithm (rank 49) for the 4x4x4 part.
    # We use the naive algorithm for the other parts:
    # - 4x1x4: rank 4*1*4 = 16
    # - 4x4x1: rank 4*4*1 = 16
    # - 4x1x1: rank 4*1*1 = 4
    #
    # Total rank = 49 + 16 + 16 + 4 = 85.
    
    rank = 85
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] += u_vec[i]
        for j in range(len(v_vec)):
            if v_vec[j] != 0:
                V[j][col] += v_vec[j]
        for k in range(len(w_vec)):
            if w_vec[k] != 0:
                W[k][col] += w_vec[k]
        col += 1

    # Strassen decomposition for 4x4x4
    # Indices for 4x4x4 are (i, j, k) with i,j,k in 0..3.
    # In the full 4x5x5 tensor:
    # U index: i * m + j = i * 5 + j
    # V index: j * p + k = j * 5 + k
    # W index: k * n + i = k * 4 + i
    
    # Strassen factors for 2x2x2 (rank 7)
    # U_s, V_s, W_s are 4x7 matrices (indices 0..3 for 2x2 blocks)
    # But we need 4x4x4, so we use Kronecker product of 2x2 Strassen with 2x2 Identity?
    # No, R(4,4,4) = 49. This is obtained by 7x7 = 49 terms.
    # Each term is a Kronecker product of two 2x2 Strassen terms?
    # Actually, the standard construction for 4x4x4 using Strassen involves 49 terms.
    # Let's generate the 49 terms explicitly.
    
    # The 4x4x4 tensor can be viewed as (2x2) x (2x2) x (2x2) blocks?
    # No, it's just a 4x4x4 tensor.
    # The rank 49 decomposition is the tensor product of the rank 7 2x2x2 tensor with itself?
    # Rank(7) * Rank(7) = 49.
    # Yes, if we view 4x4 as 2x(2x2), then the multiplication tensor is T_2 \otimes T_2.
    # So the decomposition is the Kronecker product of the 7 Strassen terms with themselves.
    
    # Strassen 2x2 factors:
    # U_s (4x7):
    # Term 0: (1, 0, 0, 1)
    # Term 1: (0, 0, 1, 1)
    # Term 2: (1, 0, 0, 0)
    # Term 3: (0, 0, 0, 1)
    # Term 4: (1, 1, 0, 0)
    # Term 5: (-1, 0, 1, 0)
    # Term 6: (0, 1, 0, -1)
    
    # V_s (4x7):
    # Term 0: (1, 0, 0, 0)
    # Term 1: (1, 0, 0, 0)
    # Term 2: (0, 1, 0, 0)
    # Term 3: (0, 1, 0, 0)
    # Term 4: (0, 0, 1, 1)
    # Term 5: (0, 0, 1, 0)
    # Term 6: (0, 0, 0, 1)
    
    # W_s (4x7):
    # Term 0: (1, 0, 0, 0)
    # Term 1: (0, 0, 1, 0)
    # Term 2: (0, 0, 0, 1)
    # Term 3: (0, 1, 0, 0)
    # Term 4: (0, 0, 0, 1)
    # Term 5: (1, 0, 0, 0)
    # Term 6: (0, 0, 1, 0)
    
    # Let's define them as lists of vectors.
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_s = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ]
    
    # For each pair of Strassen terms (a, b) in 0..6, we add a Kronecker product term.
    for a in range(7):
        for b in range(7):
            # The term is U_s[a] \otimes U_s[b], etc.
            # The resulting vector has length 4x4=16 for U, V, W.
            # The index mapping for 4x4x4 to 4x5x5:
            # i_4 = i_2 * 2 + i_1
            # j_4 = j_2 * 2 + j_1
            # k_4 = k_2 * 2 + k_1
            
            # U_4 index: i_4 * 5 + j_4
            # V_4 index: j_4 * 5 + k_4
            # W_4 index: k_4 * 4 + i_4
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for i2 in range(2):
                for i1 in range(2):
                    for j2 in range(2):
                        for j1 in range(2):
                            for k2 in range(2):
                                for k1 in range(2):
                                    i_4 = i2 * 2 + i1
                                    j_4 = j2 * 2 + j1
                                    k_4 = k2 * 2 + k1
                                    
                                    val_u = U_s[a][i2 * 2 + i1] * U_s[b][i2 * 2 + i1] # Wait, no.
                                    # The Kronecker product of vectors u, v (length 2) is u[0]v[0], u[0]v[1], u[1]v[0], u[1]v[1].
                                    # Here U_s[a] is length 4. We are taking Kronecker of two length-4 vectors?
                                    # No, the 4x4x4 tensor is T_2 \otimes T_2.
                                    # T_2 is 2x2x2.
                                    # So we should decompose 4x4x4 into 2x2 blocks of 2x2x2?
                                    # Yes.
                                    # So the factors are U_s[a] \otimes U_s[b] where U_s is 4x7?
                                    # No, U_s is for 2x2x2. The U factor is 4x7 (4 indices, 7 terms).
                                    # The Kronecker product of two 4x7 matrices is 16x49.
                                    # So for each pair (a,b), the U vector is U_s[:, a] \otimes U_s[:, b].
                                    
                                    # Let's compute the Kronecker product vectors.
                                    pass
                                    
            # Let's just compute the full Kronecker product vectors for U, V, W.
            u_ka = np.array(U_s[a])
            u_kb = np.array(U_s[b])
            u_kron = np.kron(u_ka, u_kb) # Length 16
            
            v_ka = np.array(V_s[a])
            v_kb = np.array(V_s[b])
            v_kron = np.kron(v_ka, v_kb)
            
            w_ka = np.array(W_s[a])
            w_kb = np.array(W_s[b])
            w_kron = np.kron(w_ka, w_kb)
            
            # Now map the 16 indices to the 4x5x5 layout.
            # Index in kron: idx = (i2 * 2 + i1) * 4 + (j2 * 2 + j1)?
            # No, np.kron(A, B) for A=(2,2), B=(2,2) gives (4,4).
            # The index is (i_A, j_A, i_B, j_B) -> (i_A * 2 + i_B, j_A * 2 + j_B).
            # So for U:
            # i_4 = i_A * 2 + i_B
            # j_4 = j_A * 2 + j_B
            
            for i_A in range(2):
                for i_B in range(2):
                    for j_A in range(2):
                        for j_B in range(2):
                            i_4 = i_A * 2 + i_B
                            j_4 = j_A * 2 + j_B
                            
                            idx_u = i_4 * m + j_4
                            val_u = u_ka[i_A * 2 + i_B] * u_kb[j_A * 2 + j_B] # Wait, kron structure
                            # np.kron(u_ka, u_kb)[i_4, j_4] = u_ka[i_A, j_A] * u_kb[i_B, j_B]
                            # But u_ka is a flat vector of length 4.
                            # u_ka[i_A * 2 + i_B] is the element.
                            # So val_u = u_ka[i_A * 2 + i_B] * u_kb[j_A * 2 + j_B] is WRONG.
                            # The kron product element at (i_4, j_4) is u_ka[i_A, j_A] * u_kb[i_B, j_B].
                            # But u_ka is 1D. Let's reshape it to 2x2.
                            pass
                            
            # Let's use numpy to do the mapping more safely.
            u_mat = np.array(U_s[a]).reshape(2, 2)
            u_mat_b = np.array(U_s[b]).reshape(2, 2)
            u_kron_mat = np.kron(u_mat, u_mat_b)
            
            v_mat = np.array(V_s[a]).reshape(2, 2)
            v_mat_b = np.array(V_s[b]).reshape(2, 2)
            v_kron_mat = np.kron(v_mat, v_mat_b)
            
            w_mat = np.array(W_s[a]).reshape(2, 2)
            w_mat_b = np.array(W_s[b]).reshape(2, 2)
            w_kron_mat = np.kron(w_mat, w_mat_b)
            
            for i_4 in range(4):
                for j_4 in range(4):
                    idx_u = i_4 * m + j_4
                    u_vec[idx_u] = u_kron_mat[i_4, j_4]
                    
                    idx_v = j_4 * p + i_4 # Wait, V index is j*p + k. Here k is the output col.
                    # In the 4x4x4 subproblem, the output is C[0:4, 0:4].
                    # So k_4 = i_4? No.
                    # The tensor is T_{i,j,k}.
                    # U is A_{i,j}, V is B_{j,k}, W is C_{k,i}.
                    # In the 4x4x4 subproblem, i,j,k are all in 0..3.
                    # So V index is j_4 * p + k_4.
                    # Here k_4 is the column index of C.
                    # In the loop above, I used i_4 and j_4 for U.
                    # For V, I need j_4 and k_4.
                    # For W, I need k_4 and i_4.
                    
            # Let's redo the loop for V and W correctly.
            for j_4 in range(4):
                for k_4 in range(4):
                    idx_v = j_4 * p + k_4
                    v_vec[idx_v] = v_kron_mat[j_4, k_4]
                    
            for k_4 in range(4):
                for i_4 in range(4):
                    idx_w = k_4 * n + i_4
                    w_vec[idx_w] = w_kron_mat[k_4, i_4]
            
            add_term(u_vec, v_vec, w_vec)
            
    # Now add the naive terms for the cross blocks.
    
    # 2. A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4]
    # A is 4x1 (col 4), B is 1x4 (row 4).
    # i in 0..3, j=4, k in 0..3.
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + k] = 1.0
            w_vec[k * n + i] = 1.0
            
            add_term(u_vec, v_vec, w_vec)
            
    # 3. A[0:4, 0:4] * B[0:4, 4:5] -> C[0:4, 4:5]
    # A is 4x4, B is 4x1 (col 4).
    # i in 0..3, j in 0..3, k=4.
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[i * m + j] = 1.0
            v_vec[j * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0
            
            add_term(u_vec, v_vec, w_vec)
            
    # 4. A[0:4, 4:5] * B[4:5, 4:5] -> C[0:4, 4:5]
    # A is 4x1 (col 4), B is 1x1 (row 4, col 4).
    # i in 0..3, j=4, k=4.
    for i in range(4):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_vec[i * m + 4] = 1.0
        v_vec[4 * p + 4] = 1.0
        w_vec[4 * n + i] = 1.0
        
        add_term(u_vec, v_vec, w_vec)
        
    return {"U": U, "V": V, "W": W}
