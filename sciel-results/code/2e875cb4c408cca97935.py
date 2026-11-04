import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0
    
    # Kronecker product decomposition: M_6 = M_3 ⊗ M_2
    # 3x3 schoolbook (rank 9) × 2x2 Strassen (rank 7) = rank 63
    rank = 63
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen coefficient tables
    # For each Strassen term s, define:
    # a_coeffs: dict mapping (row, col) in 2x2 A-block to coefficient
    # b_coeffs: dict mapping (row, col) in 2x2 B-block to coefficient  
    # c_coeffs: dict mapping (row, col) in 2x2 C-block to coefficient
    
    strassen_a = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): 1},      # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},      # M5
        {(1,0): 1, (0,0): -1},     # M6
        {(0,1): 1, (1,1): -1},     # M7
    ]
    strassen_b = [
        {(0,0): 1, (1,1): 1},      # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},     # M3
        {(1,0): 1, (0,0): -1},     # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},      # M6
        {(1,0): 1, (1,1): 1},      # M7
    ]
    strassen_c = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): -1},     # M2
        {(0,1): 1, (1,1): 1},      # M3
        {(0,0): 1, (1,0): 1},      # M4
        {(0,0): -1, (0,1): 1},     # M5
        {(1,1): 1},                 # M6
        {(0,0): 1},                 # M7
    ]
    
    # 3x3 schoolbook: C[i,j] = sum_k A[i,k] * B[k,j]
    # For each (i,k,j) in 3x3 grid, we have one schoolbook term
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # A block [i,k]: rows 2*i..2*i+1, cols 2*k..2*k+1
                # B block [k,j]: rows 2*k..2*k+1, cols 2*j..2*j+1
                # C block [i,j]: rows 2*i..2*i+1, cols 2*j..2*j+1
                
                for s in range(7):
                    # A-side coefficients
                    for (ar, ac), coeff in strassen_a[s].items():
                        # A[i,k] block entry (ar, ac) -> global index
                        global_row = 2*i + ar
                        global_col = 2*k + ac
                        U[global_row * m + global_col, idx] += coeff
                    
                    # B-side coefficients
                    for (br, bc), coeff in strassen_b[s].items():
                        global_row = 2*k + br
                        global_col = 2*j + bc
                        V[global_row * p + global_col, idx] += coeff
                    
                    # C-side coefficients (trace convention: W[col*n + row, rank])
                    for (cr, cc), coeff in strassen_c[s].items():
                        global_row = 2*i + cr
                        global_col = 2*j + cc
                        W[global_col * n + global_row, idx] += coeff
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Quick verification with small test
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-3, 3, (n, m)).astype(float)
        B_test = np.random.randint(-3, 3, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float).reshape(n, m, rank)
        V_arr = V.astype(float).reshape(m, p, rank)
        W_arr = W.astype(float).reshape(p, n, rank)
        
        prod = np.einsum('ikt,kjt->ijt', U_arr, V_arr)
        C_computed = np.einsum('ijt,jit->ij', prod, W_arr)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
