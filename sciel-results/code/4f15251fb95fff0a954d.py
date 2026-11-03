import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Structural approach: Maximum-packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block uses 7 rank-1 terms to cover 8 (i,j,k) tensor positions,
    saving 1 rank compared to schoolbook. We find the maximum independent set
    of non-overlapping Strassen blocks, then fill the rest with schoolbook.
    
    With n=3, m=5, p=7:
    - Possible I-pairs: {0,1}, {1,2}  (2 choices)
    - Possible J-pairs: 4 adjacent pairs (4 choices)
    - Possible K-pairs: 5 adjacent pairs (5 choices)
    - Total possible blocks: 2*4*5 = 40
    
    Rank = 105 - K where K = number of Strassen blocks packed.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

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

        # M1 = (a11+a22)*(b11+b22) -> W: +1 at c11, +1 at c22
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11 -> W: +1 at c21, -1 at c22
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22) -> W: +1 at c12, +1 at c22
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11) -> W: +1 at c11, +1 at c21
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22 -> W: -1 at c11, +1 at c12
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12) -> W: +1 at c22
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22) -> W: +1 at c11
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def get_block_cells(i1, i2, j1, j2, k1, k2):
        """Return frozenset of (i,j,k) positions covered by this block."""
        cells = set()
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    cells.add((i, j, k))
        return frozenset(cells)

    # Enumerate all possible 2x2x2 Strassen blocks
    i_pairs = [(i, i+1) for i in range(n-1)]  # (0,1), (1,2)
    j_pairs = [(j, j+1) for j in range(m-1)]  # (0,1),(1,2),(2,3),(3,4)
    k_pairs = [(k, k+1) for k in range(p-1)]  # (0,1),(1,2),(2,3),(3,4),(4,5)

    # Build list of all possible blocks with their cell sets
    all_blocks = []
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                cells = get_block_cells(i1, i2, j1, j2, k1, k2)
                all_blocks.append(((i1, i2, j1, j2, k1, k2), cells))

    block_count = len(all_blocks)
    
    # Precompute conflict masks using integers for speed
    # conflict_mask[i] is an integer where bit j is 1 if block i conflicts with block j
    conflict_masks = [0] * block_count
    for a in range(block_count):
        for b in range(a+1, block_count):
            if all_blocks[a][1] & all_blocks[b][1]:
                conflict_masks[a] |= (1 << b)
                conflict_masks[b] |= (1 << a)

    def find_max_independent_set():
        """
        Fast Branch and Bound for Maximum Independent Set using integer bitmasks.
        """
        best_set = []
        best_size = 0
        
        # Precompute neighbors for each block (for quick removal)
        # We use the conflict_masks directly.
        
        def bbs(remaining_mask, current_size, current_path):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
                
            if remaining_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_path)
                return

            # Upper bound: count bits in remaining_mask
            # If current_size + bit_count <= best_size, prune
            if current_size + remaining_mask.bit_count() <= best_size:
                return
                
            # Heuristic: Pick a vertex with high degree in the remaining graph
            # To keep it simple and fast, we can just pick the lowest bit (index 0)
            # or the one with max conflicts.
            # Let's pick the vertex with the most conflicts among remaining to prune faster.
            # But calculating max conflict is O(N).
            # Given N=40, it's fine.
            
            # Find vertex v in remaining with max degree in remaining subgraph
            max_deg = -1
            v = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                deg = (conflict_masks[idx] & remaining_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp_mask ^= lsb
            
            # Branch 1: Include v
            current_path.append(v)
            new_remaining = remaining_mask & ~(conflict_masks[v] | (1 << v))
            bbs(new_remaining, current_size + 1, current_path)
            current_path.pop()
            
            # Branch 2: Exclude v
            new_remaining2 = remaining_mask & ~(1 << v)
            bbs(new_remaining2, current_size, current_path)

        initial_mask = (1 << block_count) - 1
        bbs(initial_mask, 0, [])
        return best_set

    selected_indices = find_max_independent_set()

    # Track covered cells
    covered = set()
    for idx in selected_indices:
        block_params, cells = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered.update(cells)

    # Add schoolbook terms for uncovered (i,j,k) triples
    timeout_flag = False
    for i in range(n):
        if timeout_flag:
            break
        for j in range(m):
            if timeout_flag:
                break
            for k in range(p):
                if time.time() > deadline:
                    timeout_flag = True
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
