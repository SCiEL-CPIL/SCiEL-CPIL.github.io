import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, though this is instant
    
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = np.zeros((u_dim, rank), dtype=int)
    V = np.zeros((v_dim, rank), dtype=int)
    W = np.zeros((w_dim, rank), dtype=int)
    
    # Strassen terms for 2x2x2
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # The 7 Strassen products:
    # 1: (a00+a11)(b00+b11) -> c00, c11
    # 2: (a10+a11)b00 -> c10, c11
    # 3: a00(b01-b11) -> c01, c11
    # 4: a11(b10-b00) -> c00, c10
    # 5: (a00+a01)b11 -> c01, c00
    # 6: (a10-a00)(b00+b01) -> c11
    # 7: (a01-a11)(b10+b11) -> c00
    
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(1,0,1), (0,0,-1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    # Two blocks: k in {0,1} and k in {2,3}
    for block_idx, k_offset in enumerate([0, 2]):
        col_base = block_idx * 7
        
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(strassen_terms):
            col = col_base + t_idx
            
            for i, j, val in u_entries:
                u_idx = i * m + j
                U[u_idx][col] += val
            
            for j, k, val in v_entries:
                k_local = k + k_offset
                v_idx = j * p + k_local
                V[v_idx][col] += val
            
            for k, i, val in w_entries:
                k_local = k + k_offset
                w_idx = k_local * n + i
                W[w_idx][col] += val

    # Convert to lists as per interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
