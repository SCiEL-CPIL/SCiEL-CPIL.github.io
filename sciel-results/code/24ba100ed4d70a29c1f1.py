import time
import numpy as np

def _trivial_decomp(n, m, p, i_map, j_map, k_map):
    """
    Generate U, V, W for trivial decomposition of M_{n,m,p} mapped to global indices.
    i_map: list of global i indices for local i in 0..n-1
    j_map: list of global j indices for local j in 0..m-1
    k_map: list of global k indices for local k in 0..p-1
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Local indices
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                
                # Global indices
                gi = i_map[i]
                gj = j_map[j]
                gk = k_map[k]
                
                # Global U row index: gi * m_global + gj? 
                # Wait, the global U shape is (n_global * m_global, rank).
                # The row index in global U is gi * m_global + gj.
                # But we are returning local U, V, W which will be embedded into global.
                # Actually, it's easier to return the local U, V, W and then embed them.
                # But the local U, V, W are already in the local coordinate system.
                # We need to map the rows to global rows.
                
                U[ui][col] = 1
                V[vj][col] = 1
                W[wk][col] = 1
                col += 1
    return U, V, W

def _embed(U_local, V_local, W_local, U_global, V_global, W_global, 
           n_local, m_local, p_local, 
           i_map, j_map, k_map, 
           n_global, m_global, p_global, 
           offset):
    """
    Embed local U, V, W into global U, V, W at column offset.
    """
    rank_local = n_local * m_local * p_local
    
    # Map U rows
    for i in range(n_local):
        for j in range(m_local):
            local_row = i * m_local + j
            global_row = i_map[i] * m_global + j_map[j]
            for r in range(rank_local):
                U_global[global_row][offset + r] += U_local[local_row][r]
                
    # Map V rows
    for j in range(m_local):
        for k in range(p_local):
            local_row = j * p_local + k
            global_row = j_map[j] * p_global + k_map[k]
            for r in range(rank_local):
                V_global[global_row][offset + r] += V_local[local_row][r]
                
    # Map W rows
    for k in range(p_local):
        for i in range(n_local):
            local_row = k * n_local + i
            global_row = k_map[k] * n_global + i_map[i]
            for r in range(rank_local):
                W_global[global_row][offset + r] += W_local[local_row][r]

def _construct():
    n, m, p = 4, 5, 5
    
    # Part 1: M_{4,4,4} trivial
    n1, m1, p1 = 4, 4, 4
    i_map1 = [0, 1, 2, 3]
    j_map1 = [0, 1, 2, 3]
    k_map1 = [0, 1, 2, 3]
    U1, V1, W1 = _trivial_decomp(n1, m1, p1, i_map1, j_map1, k_map1)
    
    # Part 2: M_{4,4,1} trivial
    n2, m2, p2 = 4, 4, 1
    i_map2 = [0, 1, 2, 3]
    j_map2 = [0, 1, 2, 3]
    k_map2 = [4] # local k=0 maps to global k=4
    U2, V2, W2 = _trivial_decomp(n2, m2, p2, i_map2, j_map2, k_map2)
    
    # Part 3: M_{4,1,5} trivial
    n3, m3, p3 = 4, 1, 5
    i_map3 = [0, 1, 2, 3]
    j_map3 = [4] # local j=0 maps to global j=4
    k_map3 = [0, 1, 2, 3, 4]
    U3, V3, W3 = _trivial_decomp(n3, m3, p3, i_map3, j_map3, k_map3)
    
    total_rank = (n1*m1*p1) + (n2*m2*p2) + (n3*m3*p3)
    
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    _embed(U1, V1, W1, U_global, V_global, W_global, n1, m1, p1, i_map1, j_map1, k_map1, n, m, p, offset)
    offset += n1*m1*p1
    
    _embed(U2, V2, W2, U_global, V_global, W_global, n2, m2, p2, i_map2, j_map2, k_map2, n, m, p, offset)
    offset += n2*m2*p2
    
    _embed(U3, V3, W3, U_global, V_global, W_global, n3, m3, p3, i_map3, j_map3, k_map3, n, m, p, offset)
    
    return {"U": U_global, "V": V_global, "W": W_global}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    result = _construct()
    
    if time.time() > deadline:
        return result
    
    return result
