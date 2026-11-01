"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2 blocks
    # A indices: u = 2*i + j  (i=row, j=col of A)
    # B indices: v = 14*j + k (j=row, k=col of B)
    # C indices: w = 2*k + i  (k=col, i=row of C)
    
    # We have 7 blocks of 2x2 multiplication
    # Block q covers k = 2q and k = 2q+1
    for q in range(7):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # B indices
        v_B00 = 14 * 0 + k0  # b_0, k0
        v_B01 = 14 * 0 + k1  # b_0, k1
        v_B10 = 14 * 1 + k0  # b_1, k0
        v_B11 = 14 * 1 + k1  # b_1, k1
        
        # C indices
        w_C00 = 2 * k0 + 0   # c_0, k0
        w_C01 = 2 * k1 + 0   # c_0, k1
        w_C10 = 2 * k0 + 1   # c_1, k0
        w_C11 = 2 * k1 + 1   # c_1, k1
        
        # A indices
        u_A00 = 2 * 0 + 0    # a_00
        u_A01 = 2 * 0 + 1    # a_01
        u_A10 = 2 * 1 + 0    # a_10
        u_A11 = 2 * 1 + 1    # a_11
        
        col_base = q * 7
        
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        c = col_base + 0
        U[u_A00][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1
        W[w_C11][c] += 1
        
        # M2 = (A10 + A11)B00
        # C00 -= M2, C10 += M2
        c = col_base + 1
        U[u_A10][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        W[w_C00][c] -= 1
        W[w_C10][c] += 1
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 -= M3
        c = col_base + 2
        U[u_A00][c] += 1
        V[v_B01][c] += 1
        V[v_B11][c] -= 1
        W[w_C01][c] += 1
        W[w_C11][c] -= 1
        
        # M4 = A11(B10 - B00)
        # C00 -= M4, C10 += M4
        c = col_base + 3
        U[u_A11][c] += 1
        V[v_B10][c] += 1
        V[v_B00][c] -= 1
        W[w_C00][c] -= 1
        W[w_C10][c] += 1
        
        # M5 = (A00 + A01)B11
        # C00 += M5, C01 += M5
        c = col_base + 4
        U[u_A00][c] += 1
        U[u_A01][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1
        W[w_C01][c] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        c = col_base + 5
        U[u_A10][c] += 1
        U[u_A00][c] -= 1
        V[v_B00][c] += 1
        V[v_B01][c] += 1
        W[w_C11][c] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # C11 += M7
        c = col_base + 6
        U[u_A01][c] += 1
        U[u_A11][c] -= 1
        V[v_B10][c] += 1
        V[v_B11][c] += 1
        W[w_C11][c] += 1

    # Verify correctness with a random test
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(-10, 10, (n, m)).astype(float)
    B_test = np.random.randint(-10, 10, (m, p)).astype(float)
    C_expected = A_test @ B_test
    
    # Build factor matrices as numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape A and B to vectors
    A_vec = A_test.reshape(n * m)  # index: 2*i + j
    B_vec = B_test.reshape(m * p)  # index: 14*j + k
    
    # C_vec[w] = sum_r W[w,r] * (sum_u U[u,r] * A_vec[u]) * (sum_v V[v,r] * B_vec[v])
    C_vec = np.zeros(p * n)
    for r in range(rank):
        u_prod = np.dot(U_np[:, r], A_vec)
        v_prod = np.dot(V_np[:, r], B_vec)
        C_vec += W_np[:, r] * u_prod * v_prod
    
    C_computed = C_vec.reshape(n, p)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Decomposition is invalid; fall back to trivial rank-56
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = 2 * i + j
                    v_idx = 14 * j + k
                    w_idx = 2 * k + i
                    U[u_idx][idx] = 1
                    V[v_idx][idx] = 1
                    W[w_idx][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
