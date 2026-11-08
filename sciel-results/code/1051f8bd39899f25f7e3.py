import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen rank for 6x6 is 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions: (row_offset, col_offset)
    blocks = {
        0: (0, 0), # 00
        1: (0, 3), # 01
        2: (3, 0), # 10
        3: (3, 3)  # 11
    }
    
    # Strassen terms
    # X: list of (block_id, sign)
    # Y: list of (block_id, sign)
    # C: list of (block_id, sign)
    terms = [
        { 'X': [(0, 1), (3, 1)], 'Y': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] }, # M1
        { 'X': [(2, 1), (3, 1)], 'Y': [(0, 1)],       'C': [(2, 1), (3, -1)] }, # M2
        { 'X': [(0, 1)],       'Y': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] }, # M3
        { 'X': [(3, 1)],       'Y': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] }, # M4
        { 'X': [(0, 1), (1, 1)], 'Y': [(3, 1)],       'C': [(0, -1), (1, 1)] }, # M5
        { 'X': [(2, 1), (0, -1)], 'Y': [(0, 1), (1, 1)], 'C': [(3, 1)] },       # M6
        { 'X': [(1, 1), (3, -1)], 'Y': [(2, 1), (3, 1)], 'C': [(0, 1)] }        # M7
    ]
    
    col = 0
    for term in terms:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U entries
                    for b_id, sign in term['X']:
                        r_off, c_off = blocks[b_id]
                        g_r = r_off + i
                        g_c = c_off + j
                        idx_u = g_r * m + g_c
                        U[idx_u][col] += sign
                    
                    # V entries
                    for b_id, sign in term['Y']:
                        r_off, c_off = blocks[b_id]
                        g_r = r_off + j
                        g_c = c_off + k
                        idx_v = g_r * p + g_c
                        V[idx_v][col] += sign
                    
                    # W entries
                    for b_id, sign in term['C']:
                        r_off, c_off = blocks[b_id]
                        g_r = r_off + i
                        g_c = c_off + k
                        # W index is k*n + i (column-major)
                        idx_w = g_c * n + g_r
                        W[idx_w][col] += sign
                    
                    col += 1
                    if col >= rank:
                        break
                if col >= rank: break
            if col >= rank: break
        if col >= rank: break
        
    return {"U": U, "V": V, "W": W}
