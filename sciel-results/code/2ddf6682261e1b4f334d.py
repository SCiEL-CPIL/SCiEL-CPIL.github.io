import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    NTRIPLES = n * m * p  # 105
    
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    # Greedy search for maximum disjoint packing
    best_num_blocks = 0
    best_packing = []
    
    num_trials = 3000
    rng = random.Random(42)
    
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
            
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        
        used_indices = set()
        current_packing = []
        
        for idx in order:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            
            if best_num_blocks >= 14:
                break
    
    # Build column representation
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for idx, val in u_entries.items():
            u[idx] = val
        for idx, val in v_entries.items():
            v[idx] = val
        for idx, val in w_entries.items():
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

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

    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # Exact Gaussian elimination to find redundant columns
    def build_M():
        r = len(U_cols)
        M = [[0] * r for _ in range(NTRIPLES)]
        idx = 0
        for i in range(n):
            ui = i * m
            for j in range(m):
                vj = j * p
                for k in range(p):
                    wk = k * n + i
                    row = M[idx]
                    for c in range(r):
                        val = U_cols[c][ui + j] * V_cols[c][vj + k] * W_cols[c][wk]
                        row[c] = val
                    idx += 1
        return M

    def exact_pivots(M, deadline):
        rows = len(M)
        cols = len(M[0])
        A = [list(row) for row in M]
        pivot_cols = []
        cur = 0
        for c in range(cols):
            if time.time() > deadline:
                break
            if cur >= rows:
                break
            piv = -1
            for r in range(cur, rows):
                if A[r][c] != 0:
                    piv = r
                    break
            if piv == -1:
                continue
            A[cur], A[piv] = A[piv], A[cur]
            pv = A[cur][c]
            for r in range(cur + 1, rows):
                if A[r][c] != 0:
                    f = A[r][c] / pv
                    for cc in range(c, cols):
                        A[r][cc] -= f * A[cur][cc]
            pivot_cols.append(c)
            cur += 1
        return pivot_cols

    # Try to eliminate redundant columns
    M = build_M()
    pivots = exact_pivots(M, deadline)
    piv_set = set(pivots)
    
    # Keep only pivot columns (minimal rank subset)
    keep_indices = [c for c in range(len(U_cols)) if c in piv_set]
    
    U_cols = [U_cols[c] for c in keep_indices]
    V_cols = [V_cols[c] for c in keep_indices]
    W_cols = [W_cols[c] for c in keep_indices]
    
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
