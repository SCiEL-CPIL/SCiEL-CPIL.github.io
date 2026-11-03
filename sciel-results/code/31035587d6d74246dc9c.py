import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Strategy: Maximize the number of disjoint 2x2x2 Strassen blocks.
    Each block reduces rank by 1.
    We use randomized greedy search with bitmasks for efficient disjointness checking.
    We consider all possible pairs of indices for each dimension, not just adjacent ones.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible blocks
    # A block is defined by a pair of I indices, a pair of J indices, a pair of K indices
    all_blocks = []
    
    # Precompute cell masks for each block
    # Total cells = n * m * p = 105. Fits in a Python int.
    
    i_pairs = [(i1, i2) for i1 in range(n) for i2 in range(i1+1, n)]
    j_pairs = [(j1, j2) for j1 in range(m) for j2 in range(j1+1, m)]
    k_pairs = [(k1, k2) for k1 in range(p) for k2 in range(k1+1, p)]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                mask = 0
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            # Cell index in linearized 3D array (i, j, k)
                            # Index = i * m * p + j * p + k
                            cell_idx = i * m * p + j * p + k
                            mask |= (1 << cell_idx)
                all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))
    
    num_blocks = len(all_blocks)
    
    # Randomized greedy search
    best_count = 0
    best_packing = []
    
    # Number of trials: adjust based on time
    # 630 blocks, check is bitwise AND. Very fast.
    # Let's try 2000 trials.
    num_trials = 2000
    
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(all_blocks)
        current_mask = 0
        current_packing = []
        
        for block_params, mask in all_blocks:
            if (current_mask & mask) == 0:
                current_mask |= mask
                current_packing.append(block_params)
                
        if len(current_packing) > best_count:
            best_count = len(current_packing)
            best_packing = current_packing
            
            # Early exit if we hit the theoretical max
            # Max possible is floor(105/8) = 13
            if best_count >= 13:
                break

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered_mask = 0
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        # Update covered mask for schoolbook fill
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    cell_idx = i * m * p + j * p + k
                    covered_mask |= (1 << cell_idx)

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                cell_idx = i * m * p + j * p + k
                if not (covered_mask & (1 << cell_idx)):
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
