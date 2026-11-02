import numpy as np
import time

def construct():
    start_time = time.time()
    # We have 120s. Construction is < 1s. Verification is < 1s.
    # We are safe.
    
    n, m, p = 4, 4, 4
    
    # Build rank-49 decomposition using Strassen recursion
    U, V, W = _build_strassen_4x4(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 64 if something went wrong (should not happen)
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

def _build_strassen_4x4(n, m, p):
    # Strassen for 2x2
    # Returns U, V, W for 2x2x2 tensor of rank 7
    # U, V, W are lists of lists or numpy arrays
    # Shape: (2*2, 7) = (4, 7)
    
    # The 7 products of Strassen for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # We want to construct the 4x4 decomposition.
    # This is effectively the tensor product of the 2x2 decomposition with itself,
    # but with appropriate index permutations to match the 4x4 layout.
    
    # Easier: Just use the recursive logic from the prompt, which is correct.
    # I will copy the logic but ensure it's clean.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen 2x2 definitions
    # u_terms: (row, col, sign)
    # v_terms: (row, col, sign)
    # w_terms: (row, col, sign) -> Note: W index is col*n + row
    # For 2x2, W index is col*2 + row.
    
    inner_defs = [
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]
    
    # Outer Strassen 4x4 definitions (blocks of 2x2)
    outer_defs = [
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]
    
    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t
            
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff
            
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff
            
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx][r] += coeff
                        
    return U, V, W

def _verify(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Use a few random seeds
    for seed in [42, 123, 999]:
        rng = np.random.RandomState(seed)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        for r in range(rank):
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            w_col = W_arr[:, r].reshape(p, n)
            
            a_r = np.sum(u_col * A, axis=1)
            b_r = np.sum(v_col * B, axis=0)
            
            # w_col[k, i] = W[k*n+i, r]
            # C[i, k] += W[k*n+i, r] * a_r[i] * b_r[k]
            # w_col.T[i, k] = W[k*n+i, r]
            C_computed += np.outer(a_r, b_r) * w_col.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W
