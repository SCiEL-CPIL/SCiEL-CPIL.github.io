import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    start_time = time.time()
    
    R = 66
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    # Deadline check
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor and verify
    T = np.zeros((n, m, p))
    for r in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T, T_expected, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies among the 66 rank-1 tensors
    vec_dim = n * m * p
    M = np.zeros((66, vec_dim))
    for r in range(66):
        M[r, :] = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).flatten()
    
    if time.time() - start_time > 110:
        return {"U": U, "V": V, "W": W}
    
    try:
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = int(np.sum(s > tol * s[0]))
    except Exception:
        effective_rank = 66
    
    if effective_rank < 66:
        basis = Vt[:effective_rank, :]
        
        new_U = U_np @ basis.T  # (12, effective_rank)
        new_V = V_np @ basis.T  # (24, effective_rank)
        new_W = W_np @ basis.T  # (18, effective_rank)
        
        # Verify
        T_new = np.zeros((n, m, p))
        for s_idx in range(effective_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_new[i,j,k] += new_U[i*m+j, s_idx] * new_V[j*p+k, s_idx] * new_W[k*n+i, s_idx]
        
        if np.allclose(T_new, T_expected, atol=1e-8):
            U_list = [[float(new_U[i, s_idx]) for s_idx in range(effective_rank)] for i in range(n*m)]
            V_list = [[float(new_V[i, s_idx]) for s_idx in range(effective_rank)] for i in range(m*p)]
            W_list = [[float(new_W[i, s_idx]) for s_idx in range(effective_rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
