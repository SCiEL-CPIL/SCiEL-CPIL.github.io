import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Decomposition: C = A[0:2,0:2] × B[0:2,:] + A[:,2] × B[2,:]
    # Part 1: 2x2 x 2x7 via 3 Strassen blocks (2x2 x 2x2) + 1 trivial (2x2 x 2x1) = 3*7+4 = 25
    # Part 2: 2x1 x 1x7 = 14 scalar products
    # Total rank = 39
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs=None):
        nonlocal col
        if signs is None:
            signs = [1] * len(u_indices)
        for idx, s in zip(u_indices, signs):
            U[idx][col] += s
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        # A indices: A00=0, A01=1, A10=3, A11=4
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for col 6 of B (rows 0-1): 2x2 x 2x1, rank 4
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part for A[:, 2] * B[2, :]: 2x1 x 1x7, rank 14
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    # Verify correctness with a random test
    try:
        if time.time() - start_time < 10:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Test with random matrices
            rng = np.random.default_rng(42)
            A_test = rng.random((n, m))
            B_test = rng.random((m, p))
            
            # Compute C = A * B directly
            C_direct = A_test @ B_test
            
            # Compute C via decomposition: C[i,j] = sum_r U[i*m+r... wait
            # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # C[i,j] = sum_r U[i*m+j? No...
            
            # The tensor T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            # where i in [0,n), k in [0,m), j in [0,p)
            # C[i,j] = sum_k T[i,k,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        for r in range(rank):
                            C_decomp[i, j] += U_arr[i * m + k, r] * V_arr[k * p + j, r] * W_arr[j * n + i, r]
            
            if not np.allclose(C_direct, C_decomp, atol=1e-10):
                # If verification fails, return trivial decomposition
                rank_trivial = n * m * p
                U_t = [[0] * rank_trivial for _ in range(n * m)]
                V_t = [[0] * rank_trivial for _ in range(m * p)]
                W_t = [[0] * rank_trivial for _ in range(p * n)]
                idx = 0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            U_t[i * m + k][idx] = 1
                            V_t[k * p + j][idx] = 1
                            W_t[j * n + i][idx] = 1
                            idx += 1
                return {"U": U_t, "V": V_t, "W": W_t}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
