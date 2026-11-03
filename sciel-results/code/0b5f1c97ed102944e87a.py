import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_3x2x2(i0, i1, i2, j_base, k_base):
        nonlocal col
        a_idx = i0 * m + j_base
        b_idx = i0 * m + (j_base + 1)
        c_idx = i1 * m + j_base
        d_idx = i1 * m + (j_base + 1)
        e2_idx = i2 * m + j_base
        f2_idx = i2 * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i0
        q_idx = (k_base + 1) * n + i0
        r_idx = k_base * n + i1
        s_idx = (k_base + 1) * n + i1
        t_idx = k_base * n + i2
        u_idx = (k_base + 1) * n + i2
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        
        U[e2_idx][col] = 1
        V[e_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[g_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        U[e2_idx][col] = 1
        V[f_idx][col] = 1
        W[u_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[h_idx][col] = 1
        W[u_idx][col] = 1
        col += 1

    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(3):
            k_base = k_block * 2
            add_strassen_3x2x2(0, 1, 2, j_base, k_base)
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return _construct_schoolbook()
    
    # Try to reduce rank
    reduced = _try_reduce_rank(U_np, V_np, W_np, n, m, p)
    if reduced is not None:
        return reduced
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U_np, V_np, W_np, n, m, p, max_time=120.0):
    start = time.time()
    r0 = U_np.shape[1]
    
    # Try to find linear dependencies
    # For each column r, check if it can be removed
    U_list = [U_np[:, r].copy() for r in range(r0)]
    V_list = [V_np[:, r].copy() for r in range(r0)]
    W_list = [W_np[:, r].copy() for r in range(r0)]
    
    active = list(range(r0))
    
    for _ in range(20):  # Try up to 20 eliminations
        if time.time() - start > max_time - 10:
            break
        eliminated = False
        for idx in range(len(active)):
            r = active[idx]
            # Try removing column r
            new_active = [a for a in active if a != r]
            if not new_active:
                break
            
            # Check if the remaining columns still span the tensor
            # This is expensive, so skip for now
            pass
        
        if not eliminated:
            break
    
    return None

def _construct_schoolbook():
    n, m, p = 3, 4, 6
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
    return {"U": U, "V": V, "W": W}
