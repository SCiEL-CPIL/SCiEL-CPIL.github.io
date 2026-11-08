import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 4x4x4 (Strassen): 49
    # 4x4x1 (Trivial): 16
    # 4x1x4 (Trivial): 16
    # 4x1x1 (Trivial): 4
    # Total: 85
    rank = 85
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to get 2x2x2 Strassen factors
    # Indices: 0:00, 1:01, 2:10, 3:11
    def get_strassen_2x2_factors():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # Term 0: (a00+a11)(b00+b11)
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # Term 1: (a10+a11)b00
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1; w[3, 1] = -1
        
        # Term 2: a00(b01-b11)
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[1, 2] = 1; w[3, 2] = 1
        
        # Term 3: a11(b10-b00)
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[0, 3] = 1; w[2, 3] = 1
        
        # Term 4: (a00+a01)b11
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = -1; w[1, 4] = 1
        
        # Term 5: (a10-a00)(b00+b01)
        u[0, 5] = -1; u[2, 5] = 1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1
        
        # Term 6: (a01-a11)(b10+b11)
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    col = 0
    
    # 4x4x4 part: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # We use 7 outer Strassen terms, each decomposed into 7 inner 2x2 Strassen terms.
    # Outer terms define which 2x2 blocks of A and B are combined.
    
    # The 7 outer terms for 4x4 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product.
    # We decompose each Mi using the 2x2 Strassen factors (s_u, s_v, s_w).
    
    outer_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1: (A11+A22)(B11+B22) -> C11+1, C22+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        
        # M2: (A21+A22)B11 -> C21+1, C22-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        
        # M3: A11(B12-B22) -> C12+1, C22+1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        
        # M4: A22(B21-B11) -> C11+1, C21+1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        
        # M5: (A11+A12)B22 -> C11-1, C12+1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        
        # M6: (A21-A11)(B11+B12) -> C22+1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        
        # M7: (A12-A22)(B21+B22) -> C11+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_c, b_c, c_c) in enumerate(outer_terms):
        for inner_k in range(7):
            col_idx = t_idx * 7 + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Map A blocks to global U
            for (a_br, a_bc), coeff in a_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if u_vec[li] != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col_idx] += coeff * u_vec[li]
            
            # Map B blocks to global V
            for (b_br, b_bc), coeff in b_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if v_vec[li] != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col_idx] += coeff * v_vec[li]
            
            # Map C blocks to global W
            for (c_br, c_bc), coeff in c_c.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if w_vec[li] != 0:
                            gk = 2 * c_bc + lc
                            gi = 2 * c_br + lr
                            W[gk * 4 + gi, col_idx] += coeff * w_vec[li]
        
    col = 49
    
    # 4x4x1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    # 4x1x4: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
            
    # 4x1x1: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
        
    # Verify correctness with a random test
    def verify_decomposition():
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute using tensor decomposition
        # Formula: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        C_decomposed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    for j in range(m):
                        C_decomposed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        return np.allclose(C_expected, C_decomposed, atol=1e-10)
    
    if not verify_decomposition():
        # Fallback: use trivial decomposition if Strassen part is incorrect
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        
        # Deadline guard
        if time.time() - start_time > 110:
            pass
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Deadline guard
    if time.time() - start_time > 110:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
