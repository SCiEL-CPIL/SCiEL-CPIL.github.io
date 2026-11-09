import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 130
    
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        terms = U_arr[u_indices] * V_arr[v_indices] * W_arr[w_indices]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Rank-25: M_{2,2,5} (rank 15) + M_{2,1,5} (rank 10)
    R = 25
    U = [[0] * R for _ in range(N_U)]
    V = [[0] * R for _ in range(N_V)]
    W = [[0] * R for _ in range(N_W)]
    
    # M_{2,2,5} decomposition: C[i,t] = A[i,0]*B[0,t] + A[i,1]*B[1,t] for i=0,1, t=0..4
    # Rank 15 = 3*5. Use Strassen-style:
    # M1 = (A00+A11)*(B0+B1) -> 1x5
    # M2 = (A10+A11)*B0 -> 1x5
    # M3 = A00*(B0-B1) -> 1x5
    # Then:
    # C0 = M1 + M3 - M2 + M4 + M5... 
    # 
    # Actually, the correct formula for M_{2,2,q} with rank 3q:
    # M1 = (A00+A11) * (B0+B1)
    # M2 = (A10+A11) * B0
    # M3 = A00 * (B0-B1)
    # C0 = M1 + M3 - M2 + (A01-B11)*B1... no this is getting too complex.
    
    # Let me just use the verified 28-rank decomposition.
    R = 28
    U = [[0] * R for _ in range(N_U)]
    V = [[0] * R for _ in range(N_V)]
    W = [[0] * R for _ in range(N_W)]
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        a11 = a_rows[0] * m + a_cols[0]
        a12 = a_rows[0] * m + a_cols[1]
        a21 = a_rows[1] * m + a_cols[0]
        a22 = a_rows[1] * m + a_cols[1]
        b11 = b_rows[0] * p + b_cols[0]
        b12 = b_rows[0] * p + b_cols[1]
        b21 = b_rows[1] * p + b_cols[0]
        b22 = b_rows[1] * p + b_cols[1]
        c11 = c_cols[0] * n + c_rows[0]
        c12 = c_cols[1] * n + c_rows[0]
        c21 = c_cols[0] * n + c_rows[1]
        c22 = c_cols[1] * n + c_rows[1]
        c = start_col
        U[a11][c] += 1; U[a22][c] += 1
        V[b11][c] += 1; V[b22][c] += 1
        W[c11][c] += 1; W[c22][c] += 1
        c += 1
        U[a21][c] += 1; U[a22][c] += 1
        V[b11][c] += 1
        W[c21][c] += 1; W[c22][c] -= 1
        c += 1
        U[a11][c] += 1
        V[b12][c] += 1; V[b22][c] -= 1
        W[c12][c] += 1; W[c22][c] += 1
        c += 1
        U[a22][c] += 1
        V[b21][c] += 1; V[b11][c] -= 1
        W[c11][c] += 1; W[c21][c] += 1
        c += 1
        U[a11][c] += 1; U[a12][c] += 1
        V[b22][c] += 1
        W[c11][c] -= 1; W[c12][c] += 1
        c += 1
        U[a21][c] += 1; U[a11][c] -= 1
        V[b11][c] += 1; V[b12][c] += 1
        W[c22][c] += 1
        c += 1
        U[a12][c] += 1; U[a22][c] -= 1
        V[b21][c] += 1; V[b22][c] += 1
        W[c11][c] += 1
        c += 1
        return c
    
    def add_sb_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
        c = start_col
        for i in range(2):
            for k in range(2):
                ui = a_rows[i] * m + a_cols[k]
                vi = b_rows[k] * p + b_col
                wi = c_col * n + c_rows[i]
                U[ui][c] = 1
                V[vi][c] = 1
                W[wi][c] = 1
                c += 1
        return c
    
    def add_sb_2x1_1x5(a_rows, a_col, b_row, b_cols, c_rows, start_col):
        c = start_col
        for i in range(2):
            for k in range(5):
                ui = a_rows[i] * m + a_col
                vi = b_row * p + b_cols[k]
                wi = b_cols[k] * n + c_rows[i]
                U[ui][c] = 1
                V[vi][c] = 1
                W[wi][c] = 1
                c += 1
        return c
    
    col = add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    col = add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col)
    col = add_sb_2x2_2x1([0,1], [0,1], [0,1], 4, [0,1], 4, col)
    col = add_sb_2x1_1x5([0,1], 2, 2, [0,1,2,3,4], [0,1], col)
    
    assert col == R, f"Expected {R}, got {col}"
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return get_schoolbook()
