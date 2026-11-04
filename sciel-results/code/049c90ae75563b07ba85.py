import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a search to find a maximum packing of compatible 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible (J, K) blocks
    # J is a pair of indices from 0..m-1
    # K is a pair of indices from 0..p-1
    j_pairs = [(j1, j2) for j1 in range(m) for j2 in range(j1 + 1, m)]
    k_pairs = [(k1, k2) for k1 in range(p) for k2 in range(k1 + 1, p)]
    
    all_blocks = []
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            all_blocks.append(((j1, j2, k1, k2), frozenset([(j, k) for j in (j1, j2) for k in (k1, k2)])))

    # Compatibility check: Two blocks are compatible if their (J, K) sets are disjoint
    # This is equivalent to J1 & J2 == empty OR K1 & K2 == empty
    def is_compatible(block1, block2):
        j1, j2, k1, k2 = block1
        j1b, j2b, k1b, k2b = block2
        # Check J disjoint
        if j1 != j1b and j1 != j2b and j2 != j1b and j2 != j2b:
            return True
        # Check K disjoint
        if k1 != k1b and k1 != k2b and k2 != k1b and k2 != k2b:
            return True
        return False

    # Greedy search with local improvement
    best_packing = []
    best_size = 0
    
    num_trials = 1000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        random.shuffle(all_blocks)
        current_packing = []
        
        for block in all_blocks:
            # Check if compatible with all in current_packing
            compatible = True
            for existing in current_packing:
                if not is_compatible(block[0], existing[0]):
                    compatible = False
                    break
            if compatible:
                current_packing.append(block)
        
        if len(current_packing) > best_size:
            best_size = len(current_packing)
            best_packing = current_packing
            
            # Local search: try to swap
            # For simplicity, just keep the best greedy result for now
            # A more complex local search could be added here if time permits
            
            if best_size >= 8: # Likely max
                break

    # Construct U, V, W
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
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Use the best packing found
    # We can use the same i-pair for all blocks, e.g., (0, 1)
    covered = set()
    for block in best_packing:
        j1, j2, k1, k2 = block[0]
        i1, i2 = 0, 1
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
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
