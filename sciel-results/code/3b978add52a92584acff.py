import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks
    # A is 4x4, B is 4x6, C is 6x4
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    
    # Each C block is sum of two 2x2 products:
    # C[i1, k1] = A[i1, j1] * B[j1, k1] + A[i1, j1+1] * B[j1+1, k1]
    # where i1 in {0,1}, j1 in {0,1}, k1 in {0,1,2}
    
    # For each 2x2x2 product, use Strassen (7 terms)
    # Total: 2 (i1) * 2 (j1) * 3 (k1) * 2 (sum over j) * 7 (Strassen) = 168 terms
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for each C block C[i1, k1], we need:
    # C[i1, k1] = A[i1 block, j1 block] * B[j1 block, k1 block] + A[i1 block, j1+1 block] * B[j1+1 block, k1 block]
    
    # So for each (i1, k1), we have 2 products, each needing 7 Strassen terms = 14 terms
    # Total: 2 * 3 * 14 = 84 terms
    
    # Let's build the rank-1 terms
    rank_terms = []
    
    # Strassen terms for 2x2x2:
    # M1 = (X00+X11)(Y00+Y11) -> contributes to Z00(+1), Z11(+1)
    # M2 = (X10+X11)Y00 -> contributes to Z10(+1), Z11(-1)
    # M3 = X00(Y01-Y11) -> contributes to Z01(+1), Z11(+1)
    # M4 = X11(Y10-Y00) -> contributes to Z00(+1), Z10(+1)
    # M5 = (X00+X01)Y11 -> contributes to Z00(-1), Z01(+1)
    # M6 = (X10-X00)(Y00+Y01) -> contributes to Z11(+1)
    # M7 = (X01-X11)(Y10+Y11) -> contributes to Z00(+1)
    
    # Define Strassen terms as (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff} for X (2x2)
    # v_dict: {(j,k): coeff} for Y (2x2)
    # w_dict: {(i,k): coeff} for Z (2x2)
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Build terms for each C block
    for i1 in range(2):  # A block row
        for k1 in range(3):  # B block col (C block col)
            # C[i1, k1] = A[i1, 0] * B[0, k1] + A[i1, 1] * B[1, k1]
            
            # Product 1: A[i1, 0] * B[0, k1]
            # X = A[i1 block, 0 block], Y = B[0 block, k1 block]
            for u_dict, v_dict, w_dict in strassen_terms:
                # Map local indices to global
                u_global = {}
                for (li, lj), coeff in u_dict.items():
                    gi = i1 * 2 + li
                    gj = 0 * 2 + lj
                    u_global[(gi, gj)] = coeff
                
                v_global = {}
                for (lj, lk), coeff in v_dict.items():
                    gj = 0 * 2 + lj
                    gk = k1 * 2 + lk
                    v_global[(gj, gk)] = coeff
                
                w_global = {}
                for (li, lk), coeff in w_dict.items():
                    gi = i1 * 2 + li
                    gk = k1 * 2 + lk
                    w_global[(gi, gk)] = coeff
                
                rank_terms.append((u_global, v_global, w_global))
            
            # Product 2: A[i1, 1] * B[1, k1]
            # X = A[i1 block, 1 block], Y = B[1 block, k1 block]
            for u_dict, v_dict, w_dict in strassen_terms:
                # Map local indices to global
                u_global = {}
                for (li, lj), coeff in u_dict.items():
                    gi = i1 * 2 + li
                    gj = 1 * 2 + lj
                    u_global[(gi, gj)] = coeff
                
                v_global = {}
                for (lj, lk), coeff in v_dict.items():
                    gj = 1 * 2 + lj
                    gk = k1 * 2 + lk
                    v_global[(gj, gk)] = coeff
                
                w_global = {}
                for (li, lk), coeff in w_dict.items():
                    gi = i1 * 2 + li
                    gk = k1 * 2 + lk
                    w_global[(gi, gk)] = coeff
                
                rank_terms.append((u_global, v_global, w_global))
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
    
    # Verify correctness with a random test
    if time.time() < DEADLINE - 10:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((p, n))
        U_arr = np.array(U).T  # shape (rank, n*m)
        V_arr = np.array(V).T  # shape (rank, m*p)
        W_arr = np.array(W).T  # shape (rank, p*n)
        
        # C[k, i] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
        # This is a bit complex to vectorize directly. Let's do it element-wise for verification.
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += W[k * n + i][r]
                    # Need to sum over j: U[i*m+j, r] * V[j*p+k, r]
                    uv_sum = 0.0
                    for j in range(m):
                        uv_sum += U[i * m + j][r] * V[j * p + k][r]
                    total += W[k * n + i][r] * uv_sum
                C_computed[k, i] = total
        
        C_expected_transposed = C_expected.T  # shape (p, n)
        
        if not np.allclose(C_computed, C_expected_transposed, atol=1e-10):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1.0
                        V[j * p + k][col] = 1.0
                        W[k * n + i][col] = 1.0
                        col += 1
    
    return {"U": U, "V": V, "W": W}
