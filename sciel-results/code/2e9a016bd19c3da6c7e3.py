import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # We search for the best arrangement of up to 4 non-overlapping 2x2x2 Strassen blocks.
    # A Strassen block at (i0, j0, k0) occupies:
    # i in {i0, i0+1}
    # j in {j0, j0+1}
    # k in {k0, k0+1}
    
    # Possible starting positions for a 2x2x2 block:
    # i0 in {0, 1} (since n=3, i0+1 < 3 => i0 < 2)
    # j0 in {0, 2, 3} is wrong. j0 in {0, 2} ? No, j0+1 < 5 => j0 in {0, 1, 2, 3}.
    # But to be non-overlapping, we need to pick a set of positions.
    
    # Let's enumerate all valid sets of 4 non-overlapping blocks.
    # A block is defined by (i0, j0, k0).
    # Max i0: n-2 = 1
    # Max j0: m-2 = 3
    # Max k0: p-2 = 3
    
    all_positions = [(i, j, k) for i in range(n-1) for j in range(m-1) for k in range(p-1)]
    
    def overlaps(pos1, pos2):
        i1, j1, k1 = pos1
        i2, j2, k2 = pos2
        # Check if ranges [i1, i1+1] and [i2, i2+1] intersect
        if not (i1+1 < i2 or i2+1 < i1):
            if not (j1+1 < j2 or j2+1 < j1):
                if not (k1+1 < k2 or k2+1 < k1):
                    return True
        return False

    # Find all valid combinations of 4 non-overlapping blocks
    best_blocks = None
    min_rank = float('inf')
    
    # To save time, we can prune. But 3*4*4 = 48 positions. 48C4 is ~190k. 
    # Checking overlap for each is fast.
    
    # Actually, we can be smarter. 
    # i must be 0 or 1. If we pick i=0, we can pick i=1 later? 
    # If we pick a block at i=0, it covers i=0,1. No other block can use i=0 or i=1.
    # Since n=3, the only valid i0 values are 0 and 1.
    # If we pick a block with i0=0, it covers indices 0 and 1.
    # If we pick a block with i0=1, it covers indices 1 and 2.
    # So we cannot mix i0=0 and i0=1 blocks because they both cover i=1.
    # Therefore, ALL blocks must have the SAME i0.
    
    # Case 1: All blocks have i0 = 0.
    # They occupy i in {0,1}.
    # We need 4 non-overlapping (j,k) pairs in a 5x5 grid.
    # Possible j0: 0, 1, 2, 3. Possible k0: 0, 1, 2, 3.
    # Block covers j in {j0, j0+1}, k in {k0, k0+1}.
    
    case_valid_blocks = []
    for i0 in [0, 1]:
        jk_positions = [(j, k) for j in range(m-1) for k in range(p-1)]
        
        def overlaps_jk(pos1, pos2):
            j1, k1 = pos1
            j2, k2 = pos2
            if not (j1+1 < j2 or j2+1 < j1):
                if not (k1+1 < k2 or k2+1 < k1):
                    return True
            return False

        # Find all sets of 4 non-overlapping jk positions
        for combo in itertools.combinations(jk_positions, 4):
            is_valid = True
            for i in range(4):
                for j in range(i+1, 4):
                    if overlaps_jk(combo[i], combo[j]):
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                blocks = [(i0, cj, ck) for cj, ck in combo]
                case_valid_blocks.append(blocks)
    
    # If we found valid blocks, pick one. They all yield rank 71.
    # We just need one to construct the matrix.
    if case_valid_blocks:
        blocks = case_valid_blocks[0]
    else:
        # Fallback to hardcoded if search fails (should not happen)
        blocks = [
            (0, 0, 0), 
            (0, 2, 0), 
            (0, 0, 3), 
            (0, 2, 3)
        ]
        
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
