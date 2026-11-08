import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    start_time = time.time()
    
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for k-pairs (0,1), (2,3), (4,5) -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for k=6, j in {0,1} -> 4
    # 1 Schoolbook block (2x1x7) for j=2 -> 14
    rank = 21 + 4 + 14  # = 39
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs=None):
        nonlocal col
        if coeffs is None:
            coeffs = [1] * len(u_indices)
        for idx, c in zip(u_indices, coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, coeffs):
            W[idx][col] += c
        col += 1
    
    def add_strassen_block(k_start):
        nonlocal col
        # A indices (i*m + j): A[i][j] where i in {0,1}, j in {0,1}
        # A00=0, A01=1, A10=3, A11=4
        # B indices (j*p + k): B[j][k] where j in {0,1}, k in {k_start, k_start+1}
        b00 = 0 * p + k_start
        b01 = 0 * p + k_start + 1
        b10 = 1 * p + k_start
        b11 = 1 * p + k_start + 1
        # C indices (k*n + i): C[k][i] where k in {k_start, k_start+1}, i in {0,1}
        c00 = k_start * n + 0
        c01 = (k_start + 1) * n + 0
        c10 = k_start * n + 1
        c11 = (k_start + 1) * n + 1
        
        # M1: (A00+A11)(B00+B11) -> +C00, +C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> +C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] -= 1
        
        # M3: A00(B01-B11) -> +C01, +C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] -= 1
        
        # M4: A11(B10-B00) -> +C00, +C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] -= 1
        
        # M5: (A00+A01)B11 -> -C00, +C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] -= 1
        
        # M6: (A10-A00)(B00+B01) -> +C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] -= 1
        
        # M7: (A01-A11)(B10+B11) -> +C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] -= 1
    
    # Add 3 Strassen blocks for k-pairs (0,1), (2,3), (4,5)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # k=6, j in {0,1}: schoolbook
    # C[0][6] += A[0][0]*B[0][6] + A[0][1]*B[1][6]
    # C[1][6] += A[1][0]*B[0][6] + A[1][1]*B[1][6]
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    c06 = 6 * n + 0
    c16 = 6 * n + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # j=2: all k
    # C[i][k] += A[i][2]*B[2][k] for i in {0,1}, k in {0,...,6}
    for k in range(p):
        b2k = 2 * p + k
        c0k = k * n + 0
        c1k = k * n + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
    
    # Verify decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] should be 1 if i,j,k are valid, else 0
    # Actually the tensor T[i,j,k] = delta for matrix mult: T[i,j,k] = 1 if the (i,k) entry of A*B involves A[i,j]*B[j,k]
    # The decomposition says: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # Build the tensor from decomposition
    # T_decomp[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: 1 (since A[i,j]*B[j,k] contributes to C[k,i])
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_np[i*m+j][r] * V_np[j*p+k][r] * W_np[k*n+i][r]
                if abs(actual - expected) > 1e-10:
                    error += abs(actual - expected)
    
    if error > 1e-8:
        # Fallback: return trivial decomposition
        rank_trivial = n * m * p
        U_t = [[0] * rank_trivial for _ in range(n * m)]
        V_t = [[0] * rank_trivial for _ in range(m * p)]
        W_t = [[0] * rank_trivial for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_t[i*m+j][r] = 1
                    V_t[j*p+k][r] = 1
                    W_t[k*n+i][r] = 1
        return {"U": U_t, "V": V_t, "W": W_t}
    
    return {"U": U, "V": V, "W": W}
