import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Block Strassen with Schoolbook 3x3 inner products
    
    # U, V, W will be lists of lists
    # Size: 36 x 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to get global index for a block entry
    # block_id: 0=(11), 1=(12), 2=(21), 3=(22)
    # r, c in 0..2
    def get_idx(block_id, r, c):
        if block_id == 0: # 11
            return 3 * r, 3 * c
        elif block_id == 1: # 12
            return 3 * r, 3 * c + 3
        elif block_id == 2: # 21
            return 3 * r + 3, 3 * c
        elif block_id == 3: # 22
            return 3 * r + 3, 3 * c + 3
        else:
            raise ValueError("Invalid block id")
    
    # Define the 7 Strassen steps
    steps = [
        {'X': [(0, 1), (3, 1)], 'Y': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'X': [(2, 1), (3, 1)], 'Y': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        {'X': [(0, 1)], 'Y': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        {'X': [(3, 1)], 'Y': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'X': [(0, 1), (1, 1)], 'Y': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        {'X': [(2, 1), (0, -1)], 'Y': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        {'X': [(1, 1), (3, -1)], 'Y': [(2, 1), (3, 1)], 'C': [(0, 1)]}
    ]
    
    for step in steps:
        if time.time() > deadline:
            break
        X_defs = step['X']
        Y_defs = step['Y']
        C_defs = step['C']
        
        for r in range(3):
            for l in range(3):
                for j in range(3):
                    if time.time() > deadline:
                        break
                    a_entries = []
                    for bid, coeff in X_defs:
                        ar, ac = get_idx(bid, r, l)
                        a_entries.append((ar * m + ac, coeff))
                        
                    b_entries = []
                    for bid, coeff in Y_defs:
                        br, bc = get_idx(bid, l, j)
                        b_entries.append((br * p + bc, coeff))
                        
                    c_entries = []
                    for bid, coeff in C_defs:
                        cr, cc = get_idx(bid, r, j)
                        c_entries.append((cc * n + cr, coeff))
                    
                    for a_idx, a_coeff in a_entries:
                        for b_idx, b_coeff in b_entries:
                            U[a_idx][col] = a_coeff
                            V[b_idx][col] = b_coeff
                            for c_idx, c_coeff in c_entries:
                                W[c_idx][col] = c_coeff * a_coeff * b_coeff
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
