import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to get global index for element (r, c) in block (i, j) of a 4x4 matrix
    # i, j in {0, 1} (0-indexed blocks)
    # r, c in {0, 1} (0-indexed within block)
    # Global row = i*2 + r, Global col = j*2 + c
    # Index = Global row * 4 + Global col
    def idx_A(i, j, r, c):
        return (i*2 + r) * 4 + (j*2 + c)
        
    # For W, the index is k*n + i.
    # k is global col, i is global row.
    def idx_W(i, j, r, c):
        # i, j are block indices. r, c are element indices.
        # Global row = i*2 + r
        # Global col = j*2 + c
        # Index = Global col * 4 + Global row
        return (j*2 + c) * 4 + (i*2 + r)

    outer_steps = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)]
        }
    ]
    
    inner_steps = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)]
        }
    ]
    
    col = 0
    for k1_idx, step1 in enumerate(outer_steps):
        A_map = {}
        for (bi, bj), coeff in step1['A']:
            for r in range(2):
                for c in range(2):
                    v_idx = (r, c)
                    g_idx = idx_A(bi, bj, r, c)
                    if v_idx not in A_map:
                        A_map[v_idx] = []
                    A_map[v_idx].append((g_idx, coeff))
                    
        B_map = {}
        for (bi, bj), coeff in step1['B']:
            for r in range(2):
                for c in range(2):
                    v_idx = (r, c)
                    g_idx = idx_A(bi, bj, r, c)
                    if v_idx not in B_map:
                        B_map[v_idx] = []
                    B_map[v_idx].append((g_idx, coeff))
                    
        for k2_idx, step2 in enumerate(inner_steps):
            for v_idx, coeff in step2['A']:
                if v_idx in A_map:
                    for g_idx, a_coeff in A_map[v_idx]:
                        U[g_idx, col] += a_coeff * coeff
                        
            for v_idx, coeff in step2['B']:
                if v_idx in B_map:
                    for g_idx, b_coeff in B_map[v_idx]:
                        V[g_idx, col] += b_coeff * coeff
                        
            p_contrib = [0, 0, 0, 0]
            if k2_idx == 0:
                p_contrib[0] += 1
                p_contrib[3] += 1
            elif k2_idx == 1:
                p_contrib[2] += 1
                p_contrib[3] -= 1
            elif k2_idx == 2:
                p_contrib[1] += 1
                p_contrib[3] += 1
            elif k2_idx == 3:
                p_contrib[0] += 1
                p_contrib[2] += 1
            elif k2_idx == 4:
                p_contrib[0] -= 1
                p_contrib[1] += 1
            elif k2_idx == 5:
                p_contrib[3] += 1
            elif k2_idx == 6:
                p_contrib[0] += 1
                
            c_block_contrib = [0, 0, 0, 0]
            if k1_idx == 0:
                c_block_contrib[0] += 1
                c_block_contrib[3] += 1
            elif k1_idx == 1:
                c_block_contrib[2] += 1
                c_block_contrib[3] -= 1
            elif k1_idx == 2:
                c_block_contrib[1] += 1
                c_block_contrib[3] += 1
            elif k1_idx == 3:
                c_block_contrib[0] += 1
                c_block_contrib[2] += 1
            elif k1_idx == 4:
                c_block_contrib[0] -= 1
                c_block_contrib[1] += 1
            elif k1_idx == 5:
                c_block_contrib[3] += 1
            elif k1_idx == 6:
                c_block_contrib[0] += 1
            
            for p_idx in range(4):
                if p_contrib[p_idx] != 0:
                    pr = p_idx // 2
                    pc = p_idx % 2
                    
                    for cb_idx in range(4):
                        if c_block_contrib[cb_idx] != 0:
                            bi = cb_idx // 2
                            bj = cb_idx % 2
                            
                            g_idx_c = idx_W(bi, bj, pr, pc)
                            
                            total_coeff = p_contrib[p_idx] * c_block_contrib[cb_idx]
                            W[g_idx_c, col] += total_coeff
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
