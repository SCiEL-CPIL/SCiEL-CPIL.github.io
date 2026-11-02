import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Split p into 4 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W for the global indices.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    # For each block b (0 to 3), we handle columns 2*b to 2*b+1 of B and C.
    # Let k_offset = 2*b.
    # Local k indices: 0, 1. Global k: k_offset, k_offset+1.
    
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen terms definitions (U_part, V_part, W_part)
    # U depends on i, j
    # V depends on j, k_local
    # W depends on k_local, i
    
    # Term 1: (a00 + a11)(b00 + b11)
    # U: i=0,j=0: 1; i=1,j=1: 1
    # V: j=0,k=0: 1; j=1,k=1: 1
    # W: k=0,i=0: 1; k=1,i=1: 1
    # Wait, W index is k*n + i.
    # For k=0, i=0 -> 0. For k=1, i=1 -> 1*2+1=3.
    
    # Let's define the 7 terms for one block.
    # Each term is a tuple of (u_indices, v_indices, w_indices)
    # u_indices: list of (i, j, val)
    # v_indices: list of (j, k_local, val)
    # w_indices: list of (k_local, i, val)
    
    terms = []
    
    # M1
    u1 = [(0, 0, 1), (1, 1, 1)]
    v1 = [(0, 0, 1), (1, 1, 1)]
    w1 = [(0, 0, 1), (1, 1, 1)]
    terms.append((u1, v1, w1))
    
    # M2
    u2 = [(1, 0, 1), (1, 1, 1)]
    v2 = [(0, 0, 1)]
    w2 = [(0, 1, 1)] # k=0, i=1 -> 0*2+1=1
    terms.append((u2, v2, w2))
    
    # M3
    u3 = [(0, 0, 1)]
    v3 = [(0, 1, 1), (1, 1, -1)]
    w3 = [(1, 0, 1)] # k=1, i=0 -> 1*2+0=2
    terms.append((u3, v3, w3))
    
    # M4
    u4 = [(1, 1, 1)]
    v4 = [(1, 0, 1), (0, 0, -1)]
    w4 = [(0, 1, 1)] # k=0, i=1 -> 1
    terms.append((u4, v4, w4))
    
    # M5
    u5 = [(0, 0, 1), (0, 1, 1)]
    v5 = [(1, 1, 1)]
    w5 = [(1, 0, 1)] # k=1, i=0 -> 2
    terms.append((u5, v5, w5))
    
    # M6
    u6 = [(1, 0, 1), (0, 0, -1)]
    v6 = [(0, 0, 1), (0, 1, 1)]
    w6 = [(1, 1, 1)] # k=1, i=1 -> 3
    terms.append((u6, v6, w6))
    
    # M7
    u7 = [(0, 1, 1), (1, 1, -1)]
    v7 = [(1, 0, 1), (1, 1, 1)]
    w7 = [(1, 0, 1)] # k=1, i=0 -> 2
    terms.append((u7, v7, w7))
    
    # Now fill U, V, W for each block
    for b in range(num_blocks):
        k_offset = b * block_size
        for t_idx in range(rank_per_block):
            col = b * rank_per_block + t_idx
            u_t, v_t, w_t = terms[t_idx]
            
            for i, j, val in u_t:
                u_idx = i * m + j
                U[u_idx][col] = val
                
            for j, k_local, val in v_t:
                k_global = k_offset + k_local
                v_idx = j * p + k_global
                V[v_idx][col] = val
                
            for k_local, i, val in w_t:
                k_global = k_offset + k_local
                w_idx = k_global * n + i
                W[w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
