import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    # Index helpers
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: {(i,j): coeff}, V: {(j,k): coeff}, W: {(i,k): coeff}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (0,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1})
    ]
    
    # Note: The W coefficients above need to be mapped to the correct C indices.
    # Let's re-verify the Strassen mapping to C indices.
    # c11 (0,0) = M1 - M4 + M5 - M7
    # c12 (0,1) = M3 + M5
    # c21 (1,0) = M2 + M4
    # c22 (1,1) = M1 - M2 + M3 + M6
    
    # Let's define the contribution of each M_t to C_{i,k}
    # M1: c11(+1), c22(+1)
    # M2: c21(+1), c22(-1)
    # M3: c12(+1), c22(+1)
    # M4: c11(-1), c21(+1)
    # M5: c11(+1), c12(+1)
    # M6: c22(+1)
    # M7: c11(-1)
    
    # Corrected Strassen terms for U, V, W contributions per rank unit t
    # U_t is the combination of A elements
    # V_t is the combination of B elements
    # W_t is the combination of C elements
    
    strassen_defs = [
        # t=0: M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # t=1: M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # t=2: M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # t=3: M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        # t=4: M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
        # t=5: M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # t=6: M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1})
    ]
    
    def add_strassen_block(k0, col_offset):
        for t in range(7):
            col = col_offset + t
            u_coeff, v_coeff, w_coeff = strassen_defs[t]
            for (i, j), c in u_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in v_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
            for (i, kl), c in w_coeff.items():
                W[C_idx(i, k0 + kl), col] += c

    def add_single_block(k, col_offset):
        # Schoolbook for 2x2 * 2x1
        # c10 = a10*b0k + a11*b1k
        # c20 = a20*b0k + a21*b1k
        # t=0: a10*b0k -> c10
        U[A_idx(0,0), col_offset+0] = 1
        V[B_idx(0,k), col_offset+0] = 1
        W[C_idx(0,k), col_offset+0] = 1
        
        # t=1: a11*b1k -> c10
        U[A_idx(0,1), col_offset+1] = 1
        V[B_idx(1,k), col_offset+1] = 1
        W[C_idx(0,k), col_offset+1] = 1
        
        # t=2: a20*b0k -> c20
        U[A_idx(1,0), col_offset+2] = 1
        V[B_idx(0,k), col_offset+2] = 1
        W[C_idx(1,k), col_offset+2] = 1
        
        # t=3: a21*b1k -> c20
        U[A_idx(1,1), col_offset+3] = 1
        V[B_idx(1,k), col_offset+3] = 1
        W[C_idx(1,k), col_offset+3] = 1

    # Build the decomposition
    col = 0
    # 6 Strassen blocks for k0 = 0, 2, 4, 6, 8, 10
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
        
    # 1 Single block for k=12
    add_single_block(12, col)
    col += 4
    
    assert col == R, f"Rank mismatch: {col} != {R}"
    
    # Verification
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    if target_indices:
        target_arr = np.array(target_indices)
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        
        U_t = U[a_idx, :]
        V_t = V[b_idx, :]
        W_t = W[c_idx, :]
        
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        
        if not np.all(np.abs(sums - 1.0) < 1e-9):
            # Fallback to schoolbook if verification fails
            R0 = n * m * p
            U0 = np.zeros((nm, R0), dtype=float)
            V0 = np.zeros((mp, R0), dtype=float)
            W0 = np.zeros((pn, R0), dtype=float)
            col_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U0[i * m + j, col_idx] = 1.0
                        V0[j * p + k, col_idx] = 1.0
                        W0[k * n + i, col_idx] = 1.0
                        col_idx += 1
            return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
