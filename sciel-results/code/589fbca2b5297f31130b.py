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
    # a_block_coeffs: dict (br, bc) -> coeff
    # b_block_coeffs: dict (br, bc) -> coeff
    # c_block_coeffs: dict (br, bc) -> coeff
    
    # The 7 outer terms for 4x4 Strassen:
    # 1. (A11+A22)(B11+B22)
    # 2. (A21+A22)B11
    # 3. A11(B12-B22)
    # 4. A22(B21-B11)
    # 5. (A11+A12)B22
    # 6. (A21-A11)(B11+B12)
    # 7. (A12-A22)(B21+B22)
    
    # Note: In standard 4x4 Strassen, the blocks are 2x2.
    # A11 is top-left, A12 top-right, A21 bottom-left, A22 bottom-right.
    # B11 top-left, etc.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Wait, the standard Strassen for 2x2x2 uses 7 products.
    # For 4x4, we apply Strassen to the 2x2 blocks.
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
        # a_coeffs: {(br, bc): coeff}
        # b_coeffs: {(br, bc): coeff}
        # c_coeffs: {(br, bc): coeff}
        
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
        
    # Deadline guard
    if time.time() - start_time > 110:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
