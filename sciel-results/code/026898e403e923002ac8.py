import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 (cols 0-3 of C) using Strassen + Trivial 2x2x2
    # Part 2: 4x4x3 (cols 4-6 of C) using Trivial
    
    rank_part1 = 7 * 8 # 56
    rank_part2 = 4 * 4 * 3 # 48
    rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to get global index for A block
    def get_A_idx(block, local_i, local_j):
        bi = block // 2
        bj = block % 2
        gi = bi * 2 + local_i
        gj = bj * 2 + local_j
        return gi * 4 + gj
        
    # Helper to get global index for B block (cols 0-3)
    def get_B_idx(block, local_i, local_j):
        # block 0: B00, 1: B01, 2: B10, 3: B11
        bi = block // 2
        bj = block % 2
        gi = bi * 2 + local_i
        gj = bj * 2 + local_j
        return gi * 7 + gj
        
    # Helper to get global index for C block (cols 0-3)
    def get_C_idx(block, local_i, local_j):
        # block 0: C00, 1: C01, 2: C10, 3: C11
        # C is 4x7, index k*4+i.
        # block 0: k in 0,1; i in 0,1
        # block 1: k in 2,3; i in 0,1
        # block 2: k in 0,1; i in 2,3
        # block 3: k in 2,3; i in 2,3
        kb = block // 2
        ib = block % 2
        gk = kb * 2 + local_i # Note: local_i in C context is k index in block?
        # Wait, C[k,i]. In Strassen, C00 corresponds to k in {0,1}, i in {0,1}.
        # My block definition:
        # C00: k in {0,1}, i in {0,1}
        # C01: k in {2,3}, i in {0,1}
        # C10: k in {0,1}, i in {2,3}
        # C11: k in {2,3}, i in {2,3}
        
        # Let's stick to previous mapping:
        # block 0: C00
        # block 1: C01
        # block 2: C10
        # block 3: C11
        
        if block == 0:
            gk = local_i
            gi = local_j
        elif block == 1:
            gk = 2 + local_i
            gi = local_j
        elif block == 2:
            gk = local_i
            gi = 2 + local_j
        else: # block 3
            gk = 2 + local_i
            gi = 2 + local_j
            
        return gk * 4 + gi

    # Strassen Terms Definition
    # Each term has:
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    terms = [
        { # M1
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        { # M2
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        { # M3
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]
        },
        { # M4
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        { # M5
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        { # M6
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        { # M7
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)]
        }
    ]
    
    start_time = time.time()
    timeout = 110.0
    
    # Trivial 2x2x2 sub-terms: (Z_k, Z_i, X_k, X_i, Y_k, Y_i)
    # Z = X Y
    # Z[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
    # Z[0,1] = X[0,0]Y[0,1] + X[0,1]Y[1,1]
    # Z[1,0] = X[1,0]Y[0,0] + X[1,1]Y[1,0]
    # Z[1,1] = X[1,0]Y[0,1] + X[1,1]Y[1,1]
    
    trivial_subterms = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 0),
        (1, 1, 1, 0, 0, 1),
        (0, 0, 0, 1, 1, 0),
        (0, 1, 0, 1, 1, 1),
        (1, 0, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 1)
    ]
    
    t_idx = 0
    for term in terms:
        if time.time() - start_time > timeout:
             # Fallback to a simpler decomposition if we run out of time?
             # But we should finish quickly.
             break
             
        for sub in trivial_subterms:
            z_k, z_i, x_k, x_i, y_k, y_i = sub
            
            # Process U (A)
            # X element (x_k, x_i) is sum of coeff * A_block[local]
            for ab_idx, coeff in term['A']:
                if coeff == 0: continue
                # local indices in A block
                la_i = x_k
                la_j = x_i
                global_a_idx = get_A_idx(ab_idx, la_i, la_j)
                U[global_a_idx, t_idx] += coeff
                
            # Process V (B)
            for bb_idx, coeff in term['B']:
                if coeff == 0: continue
                lb_i = y_k
                lb_j = y_i
                global_b_idx = get_B_idx(bb_idx, lb_i, lb_j)
                V[global_b_idx, t_idx] += coeff
                
            # Process W (C)
            for cb_idx, coeff in term['C']:
                if coeff == 0: continue
                # local indices in C block
                # z_k corresponds to k index in block
                # z_i corresponds to i index in block
                lc_k = z_k
                lc_i = z_i
                global_c_idx = get_C_idx(cb_idx, lc_k, lc_i)
                W[global_c_idx, t_idx] += coeff
                
            t_idx += 1
            
    # Part 2: Trivial for cols 4-6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                
                U[u_idx, t_idx] = 1
                V[v_idx, t_idx] = 1
                W[w_idx, t_idx] = 1
                t_idx += 1
                
                if time.time() - start_time > timeout:
                    break
            if time.time() - start_time > timeout:
                break
        if time.time() - start_time > timeout:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

