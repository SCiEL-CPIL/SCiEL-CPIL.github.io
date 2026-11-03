import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0
    
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # A indices
        a = i_base * m + j_base
        b = i_base * m + (j_base + 1)
        c = (i_base + 1) * m + j_base
        d = (i_base + 1) * m + (j_base + 1)
        
        # B indices
        e = j_base * p + k_base
        f = j_base * p + (k_base + 1)
        g = (j_base + 1) * p + k_base
        h = (j_base + 1) * p + (k_base + 1)
        
        # C indices
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h)
        U[a][col] = 1; U[d][col] = 1
        V[e][col] = 1; V[h][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c][col] = 1; U[d][col] = 1
        V[e][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a][col] = 1
        V[f][col] = 1; V[h][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d][col] = 1
        V[g][col] = 1; V[e][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a][col] = 1; U[b][col] = 1
        V[h][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c][col] = 1; U[a][col] = -1
        V[e][col] = 1; V[f][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b][col] = 1; U[d][col] = -1
        V[g][col] = 1; V[h][col] = 1
        W[p_idx][col] = 1
        col += 1
    
    # Row 0: schoolbook (24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen (42 terms)
    for k_block in range(p // 2):
        k_base = k_block * 2
        for j_block in range(m // 2):
            j_base = j_block * 2
            add_strassen(1, j_base, k_base)
    
    # Verification (optional but good for safety)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Compute result
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if it's all 1s
    if not np.allclose(result, 1.0, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
