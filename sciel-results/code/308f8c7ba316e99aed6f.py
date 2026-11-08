import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses Integer Linear Programming (if available) or a refined greedy search to find 
    a maximum packing of disjoint 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # The set of indices covered by this block
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
                            
    num_blocks_total = len(all_blocks)
    num_triples = n * m * p
    
    # Map triple to index
    triple_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_idx[(i, j, k)] = idx
                idx += 1

    # Build incidence matrix A (num_triples x num_blocks_total)
    # A[t, b] = 1 if block b covers triple t
    A = np.zeros((num_triples, num_blocks_total), dtype=np.float64)
    for b_idx, (params, indices) in enumerate(all_blocks):
        for t in indices:
            A[triple_to_idx[t], b_idx] = 1.0

    best_packing = []
    
    # Try ILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Minimize -sum(x) subject to A @ x <= 1, x in {0,1}
        c = -np.ones(num_blocks_total)
        constraints = [LinearConstraint(A, ub=np.ones(num_triples))]
        integrality = np.ones(num_blocks_total)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            # x should be 0 or 1
            for b_idx in range(num_blocks_total):
                if x[b_idx] > 0.5:
                    best_packing.append(all_blocks[b_idx][0])
            print(f"ILP found packing of size {len(best_packing)}")
        else:
            print("ILP failed, falling back to greedy")
            best_packing = []
            
    except Exception as e:
        print(f"ILP exception: {e}, falling back to greedy")
        best_packing = []

    # Fallback: Greedy search if ILP failed or found nothing
    if not best_packing:
        # Use a heuristic greedy: pick block with minimum conflicts
        # Pre-compute conflict counts
        # Conflict count for block b is number of other blocks that intersect b.
        # Or just use random restarts as in the original, but maybe more trials.
        
        best_num_blocks = 0
        
        # Heuristic: Sort blocks by "density" or just random.
        # Let's try a few random restarts.
        num_trials = 500
        for t in range(num_trials):
            if time.time() > deadline - 5.0:
                break
                
            shuffled_blocks = all_blocks[:]
            random.shuffle(shuffled_blocks)
            
            used_indices = set()
            current_packing = []
            
            for block_params, block_indices in shuffled_blocks:
                if not (block_indices & used_indices):
                    current_packing.append(block_params)
                    used_indices |= block_indices
                    
            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = current_packing[:]
                
                if best_num_blocks >= 13: # Theoretical max
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
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
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
