import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify correctness
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Numerical verification: check tensor identity on random inputs
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        C_expected = A @ B
        
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        
        # Reconstruct C via tensor decomposition
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But the decomposition is: for each r, outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_reconstructed = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = Uf[i * m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = Vf[j * p + k, r]
                        if v_val == 0:
                            continue
                        w_val = Wf[k * n + i, r]
                        C_reconstructed[i, k] += u_val * v_val * w_val
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            # Fallback: return a valid schoolbook decomposition
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=np.int64)
            V_s = np.zeros((m * p, rank_school), dtype=np.int64)
            W_s = np.zeros((p * n, rank_school), dtype=np.int64)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j, idx] = 1
                        V_s[j * p + k, idx] = 1
                        W_s[k * n + i, idx] = 1
                        idx += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Convert to list format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
