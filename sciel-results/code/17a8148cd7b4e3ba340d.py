import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # Each term is (u_map, v_map, w_map)
    # Keys are (row, col)
    laderman_terms = [
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (2,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1, (1,2): 1}, {(2,2): 1}, {(2,1): 1, (2,2): 1}),
        ({(2,2): 1}, {(2,0): 1, (0,0): -1}, {(0,2): 1, (0,0): -1}),
        ({(2,0): 1, (2,2): 1}, {(0,0): 1}, {(0,2): 1, (0,0): 1}),
        ({(0,2): 1, (2,2): 1}, {(2,0): 1}, {(0,0): 1, (0,2): 1}),
        ({(0,0): 1, (1,1): -1}, {(1,1): 1, (2,2): 1}, {(1,0): 1, (2,0): 1, (1,1): -1, (2,1): -1}),
        ({(1,1): 1, (2,2): -1}, {(2,2): 1, (0,0): 1}, {(2,1): 1, (0,1): 1, (2,2): -1, (0,2): -1}),
        ({(2,2): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,2): 1, (1,2): 1, (0,0): -1, (1,0): -1}),
        ({(0,1): 1, (1,1): 1}, {(1,2): 1}, {(2,0): 1, (2,1): 1}),
        ({(1,1): 1, (2,1): 1}, {(2,1): 1}, {(1,1): 1, (1,2): 1}),
        ({(2,1): 1, (0,1): 1}, {(1,1): 1}, {(1,2): 1, (1,0): 1}),
        ({(0,2): 1, (1,2): 1}, {(2,0): 1}, {(0,0): 1, (0,1): 1}),
        ({(1,2): 1, (2,2): 1}, {(0,2): 1}, {(2,1): 1, (2,2): 1}),
        ({(2,2): 1, (0,2): 1}, {(2,0): 1}, {(0,0): 1, (0,2): 1}),
        ({(0,0): 1, (1,0): 1}, {(0,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,0): 1, (2,0): 1}, {(1,2): 1}, {(2,0): 1, (2,1): 1}),
        ({(2,0): 1, (0,0): 1}, {(2,1): 1}, {(0,1): 1, (0,2): 1}),
        ({(0,1): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1, (2,0): 1}, {(2,1): 1}, {(2,0): 1, (2,1): 1}),
        ({(2,1): 1, (0,0): 1}, {(0,1): 1}, {(0,1): 1, (0,2): 1}),
        ({(0,2): 1, (1,0): 1}, {(2,0): 1}, {(0,0): 1, (0,1): 1}),
        ({(1,2): 1, (2,0): 1}, {(0,2): 1}, {(2,1): 1, (2,2): 1}),
    ]
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We have two blocks of B: B1 (cols 0-2) and B2 (cols 3-5)
    # For block 0 (B1), V indices are j*6 + k (k in 0..2)
    # For block 1 (B2), V indices are j*6 + (k+3) (k in 0..2)
    # W indices for block 0 are k*3 + i (k in 0..2)
    # W indices for block 1 are (k+3)*3 + i (k in 0..2)
    
    col = 0
    for block_offset in [0, 3]:
        for u_map, v_map, w_map in laderman_terms:
            for (i, j), coeff in u_map.items():
                U[i * m + j][col] += coeff
            for (j, k), coeff in v_map.items():
                # j is 0..2, k is 0..2 (local)
                # Global col in B is j * p + (k + block_offset)
                V[j * p + (k + block_offset)][col] += coeff
            for (k, i), coeff in w_map.items():
                # k is 0..2 (local), i is 0..2
                # Global row in C is (k + block_offset) * n + i
                W[(k + block_offset) * n + i][col] += coeff
            col += 1
            
    # Verify correctness
    # C = A * B
    # C[i, k_global] = sum_j A[i, j] * B[j, k_global]
    # Our decomposition:
    # C[i, k_global] = sum_t U[i, j]_t * V[j, k_global]_t * W[k_global, i]_t ? No.
    # The tensor identity is:
    # sum_t U[i, j]_t V[j, k]_t W[k, i]_t = delta_ij? No.
    # The identity is: sum_t U[i, j]_t V[j, k]_t = sum_k' W[k', i]_t * (something)?
    # Actually, the decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # = sum_t (sum_j U[i, j]_t A[i, j]) (sum_k V[j, k]_t B[j, k]) ... No.
    # It is:
    # sum_t U_t V_t W_t = Tensor
    # Where U_t is a vector in A-space, V_t in B-space, W_t in C-space.
    # U_t = sum_ij u_t_ij A_ij
    # V_t = sum_jk v_t_jk B_jk
    # W_t = sum_ki w_t_ki C_ik
    # Identity: sum_t U_t V_t = sum_ik (sum_j A_ij B_jk) C_ik ? No.
    # The identity is:
    # sum_t (sum_ij u_t_ij A_ij) (sum_jk v_t_jk B_jk) = sum_ik (sum_j A_ij B_jk) w_t_ki ? No.
    # Correct identity:
    # sum_t u_t_ij v_t_jk w_t_ki = delta_ij? No.
    # sum_t u_t_ij v_t_jk = (1/w_t_ki) * ...
    
    # Let's check the standard form:
    # C_ik = sum_j A_ij B_jk
    # sum_t (sum_ij u_t_ij A_ij) (sum_jk v_t_jk B_jk) (sum_ki w_t_ki C_ik) ? No.
    
    # The factorization is:
    # C_ik = sum_j A_ij B_jk
    # We propose:
    # sum_t (sum_ij u_t_ij A_ij) (sum_jk v_t_jk B_jk) = sum_ik C_ik w_t_ki
    # This is not quite right dimensionally.
    
    # Correct check:
    # For random A, B:
    # C = A @ B
    # LHS = sum_t (U_t * A_flat) * (V_t * B_flat)
    # This scalar product should equal sum_ik C_ik * W_t_ki ?
    # No, the decomposition is:
    # sum_t U_t[i,j] V_t[j,k] W_t[k,i] = A[i,j] B[j,k] ? No.
    
    # The tensor T_ijk = sum_t U_t[i,j] V_t[j,k] W_t[k,i]
    # This T_ijk should be 1 if it corresponds to the multiplication term?
    # No, T_ijk is the tensor of multiplication.
    # T_ijk = 1 if i,j,k are such that C_ik gets contribution from A_ij B_jk.
    # Actually, the multiplication tensor M has entries M[i, j, k] = 1.
    # Wait, M[i, j, k] is not 1 everywhere.
    # M[i, j, k] = 1 for all i, j, k?
    # C_ik = sum_j A_ij B_jk.
    # So the coefficient of A_ij B_jk in C_ik is 1.
    # So the tensor T has T[i, j, k] = 1 for all i, j, k?
    # No, T is a 3-tensor. T[i, j, k] = 1.
    # Then sum_t U_t[i] V_t[j] W_t[k] = T[i, j, k] = 1?
    # No, U is n*m, V is m*p, W is p*n.
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    # The term is U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Sum over t should be 1 for all i, j, k?
    # Yes, because C_ik = sum_j A_ij B_jk.
    # So sum_t U_t[i*m+j] V_t[j*p+k] W_t[k*n+i] = 1 for all i, j, k.
    
    # Let's verify this property.
    import numpy as np
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    # Compute C from decomposition
    C_calc = np.zeros((n, p))
    for t in range(rank):
        U_t = np.array([U[i*m+j][t] for i in range(n) for j in range(m)]).reshape(n, m)
        V_t = np.array([V[j*p+k][t] for j in range(m) for k in range(p)]).reshape(m, p)
        W_t = np.array([W[k*n+i][t] for k in range(p) for i in range(n)]).reshape(p, n)
        
        # This doesn't directly give C.
        # The identity is:
        # sum_t (A * U_t^T) * (V_t * B^T) ...
        
        # Let's use the tensor check:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if not np.isclose(s, 1.0):
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col_fb = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_fb[i * m + j][col_fb] = 1
                                V_fb[j * p + k][col_fb] = 1
                                W_fb[k * n + i][col_fb] = 1
                                col_fb += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
                    
    return {"U": U, "V": V, "W": W}
