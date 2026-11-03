import numpy as np

def construct():
    n, m, p = 3, 5, 7
    # Splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    # Precompute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Total rank
    total_rank = 0
    block_ranks = []
    for i, ns in enumerate(n_splits):
        for j, ms in enumerate(m_splits):
            for k, ps in enumerate(p_splits):
                # Determine rank of block ns x ms x ps
                if ns == 2 and ms == 2 and ps == 2:
                    r = 7
                elif ns == 2 and ms == 2 and ps == 1:
                    r = 4
                elif ns == 2 and ms == 1 and ps == 2:
                    r = 4
                elif ns == 2 and ms == 1 and ps == 1:
                    r = 2
                elif ns == 1 and ms == 2 and ps == 2:
                    r = 4
                elif ns == 1 and ms == 2 and ps == 1:
                    r = 2
                elif ns == 1 and ms == 1 and ps == 2:
                    r = 2
                elif ns == 1 and ms == 1 and ps == 1:
                    r = 1
                else:
                    r = ns * ms * ps # Fallback
                block_ranks.append((i, j, k, ns, ms, ps, r))
                total_rank += r
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (i, j, k, ns, ms, ps, r) in block_ranks:
        i_off = n_offsets[i]
        j_off = m_offsets[j]
        k_off = p_offsets[k]
        
        # Generate local U, V, W
        # U is ns*ms x r
        # V is ms*ps x r
        # W is ps*ns x r
        
        if ns == 2 and ms == 2 and ps == 2:
            # Strassen
            u_b = [[0]*r for _ in range(4)]
            v_b = [[0]*r for _ in range(4)]
            w_b = [[0]*r for _ in range(4)]
            
            # M1
            u_b[0][0] = 1; u_b[3][0] = 1
            v_b[0][0] = 1; v_b[3][0] = 1
            w_b[0][0] = 1; w_b[3][0] = 1
            
            # M2
            u_b[2][1] = 1; u_b[3][1] = 1
            v_b[0][1] = 1
            w_b[1][1] = 1; w_b[2][1] = 1
            
            # M3
            u_b[0][2] = 1
            v_b[1][2] = 1; v_b[3][2] = -1
            w_b[0][2] = 1; w_b[1][2] = 1
            
            # M4
            u_b[3][3] = 1
            v_b[2][3] = 1; v_b[0][3] = -1
            w_b[1][3] = 1; w_b[2][3] = 1
            
            # M5
            u_b[0][4] = 1; u_b[1][4] = 1
            v_b[3][4] = 1
            w_b[0][4] = -1; w_b[3][4] = 1
            
            # M6
            u_b[2][5] = 1; u_b[0][5] = -1
            v_b[0][5] = 1; v_b[1][5] = 1
            w_b[2][5] = 1; w_b[3][5] = 1
            
            # M7
            u_b[1][6] = 1; u_b[3][6] = -1
            v_b[2][6] = 1; v_b[3][6] = 1
            w_b[0][6] = 1; w_b[2][6] = -1
            
            # Map to global
            for a in range(4):
                global_u_row = (i_off + a // 2) * m + (j_off + a % 2)
                for c in range(r):
                    U[global_u_row][col_offset + c] += u_b[a][c]
            
            for a in range(4):
                global_v_row = (j_off + a // 2) * p + (k_off + a % 2)
                for c in range(r):
                    V[global_v_row][col_offset + c] += v_b[a][c]
            
            for a in range(4):
                global_w_row = (k_off + a // 2) * n + (i_off + a % 2)
                for c in range(r):
                    W[global_w_row][col_offset + c] += w_b[a][c]
            
        else:
            # Generic case for small blocks (rank = ns*ms*ps)
            # We can just use schoolbook for these small ones
            # But we determined specific ranks above.
            # For simplicity and correctness, we can use schoolbook for non-Strassen blocks
            # since their ranks are small and schoolbook is optimal for 1x1x1, 1x1x2, etc.
            # Wait, 2x2x1 rank 4 is schoolbook.
            # 2x1x2 rank 4 is schoolbook.
            # 1x2x2 rank 4 is schoolbook.
            # So we can just use schoolbook for all non-2x2x2 blocks.
            
            for a in range(ns * ms):
                ia = a // ms
                ja = a % ms
                global_u_row = (i_off + ia) * m + (j_off + ja)
                for c in range(ms * ps):
                    jb = c // ps
                    kb = c % ps
                    global_v_row = (j_off + jb) * p + (k_off + kb)
                    global_w_row = (k_off + kb) * n + (i_off + ia)
                    if a == c: # This logic is wrong for schoolbook mapping
                        pass
            
            # Correct Schoolbook mapping:
            # Each term (i, j, k) is a rank 1 component.
            # But we are summing them into a block.
            # For a block of size ns x ms x ps, the schoolbook rank is ns*ms*ps.
            # We assign r = ns*ms*ps columns.
            for c in range(r):
                # c corresponds to (ia, ja, ka)
                ia = c // (ms * ps)
                ja = (c // ps) % ms
                ka = c % ps
                
                global_u_row = (i_off + ia) * m + (j_off + ja)
                global_v_row = (j_off + ja) * p + (k_off + ka)
                global_w_row = (k_off + ka) * n + (i_off + ia)
                
                U[global_u_row][col_offset + c] = 1
                V[global_v_row][col_offset + c] = 1
                W[global_w_row][col_offset + c] = 1
        
        col_offset += r
        
    return {"U": U, "V": V, "W": W}
