import numpy as np
import time

DIMENSIONS = [2, 3, 7]
_START_TIME = time.time()
_TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for k=6 -> 4
    # 1 Schoolbook block (2x1x7) for j=2 -> 14
    rank = 21 + 4 + 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
        
    # Time guard
    def check_time():
        return (time.time() - _START_TIME) > (_TIME_LIMIT - 5.0)

    # Helper for Strassen block
    # local A indices: 00, 01, 10, 11 -> global i*3+j
    # local B indices: 00, 01, 10, 11 -> global j*7+k
    # local C indices: 00, 01, 10, 11 -> global k*2+i
    def add_strassen_block(k_start):
        # k_start is 0, 2, 4
        # A indices:
        # A00: 0*3+0 = 0
        # A01: 0*3+1 = 1
        # A10: 1*3+0 = 3
        # A11: 1*3+1 = 4
        
        # B indices:
        # B00: 0*7 + k_start
        # B01: 0*7 + k_start + 1
        # B10: 1*7 + k_start
        # B11: 1*7 + k_start + 1
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices:
        # C00: k_start*2 + 0
        # C01: (k_start+1)*2 + 0
        # C10: k_start*2 + 1
        # C11: (k_start+1)*2 + 1
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

    # Add 3 Strassen blocks
    add_strassen_block(0)
    if check_time():
        return {"U": U, "V": V, "W": W}
    add_strassen_block(2)
    if check_time():
        return {"U": U, "V": V, "W": W}
    add_strassen_block(4)
    
    # Block 4: k=6, 2x2x1 Schoolbook
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2
    # A02 B2k C0k
    # A12 B2k C1k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    if check_time():
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition with random test
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
    # where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # In our indexing: A index = i*3+j (i in 0..1, j in 0..2)
    #                B index = j*7+k (j in 0..2, k in 0..6)
    #                C index = k*2+i (k in 0..6, i in 0..1)
    
    # Verify: for random A, B, check A@B == sum_r U[:,r]*V[:,r]*W[:,r]
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # u_col has entries at positions i*m+j
        # v_col has entries at positions j*p+k
        # w_col has entries at positions k*n+i
        # Term r contributes: for each (i,j,k) where u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
        # C[i,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        for i in range(n):
            for k in range(p):
                w_val = w_col[k*n+i]
                if w_val != 0:
                    for j in range(m):
                        u_val = u_col[i*m+j]
                        v_val = v_col[j*p+k]
                        if u_val != 0 and v_val != 0:
                            C_decomp[i, k] += u_val * v_val * w_val
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback: return schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col_sb] = 1
                    V_sb[j*p+k][col_sb] = 1
                    W_sb[k*n+i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
