<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses 6 Strassen 2x2x2 blocks (rank 7 each) plus schoolbook for the rest.
    Total rank: 6*7 + (105 - 6*8) = 42 + 57 = 99.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts: {index: value}"""
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
        """Add 7 Strassen rank-1 terms for a 2x2x2 block.
        
        A submatrix: rows i1,i2, cols j1,j2
        B submatrix: rows j1,j2, cols k1,k2
        C submatrix: rows i1,i2, cols k1,k2
        
        U indices: i*m+j
        V indices: j*p+k
        W indices: k*n+i
        """
        # A positions
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        # B positions
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        # C positions
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

    # Define 6 Strassen blocks
    # Using i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for (i1, i2, j1, j2, k1, k2) in blocks:
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
    
    # Convert lists to proper matrix format
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    # Ensure correct shapes
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
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Searches for maximum number of non-overlapping 2x2x2 Strassen blocks,
    then uses schoolbook for remaining entries.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2:
                        continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2:
                                continue
                            all_blocks.append((i1, i2, j1, j2, k1, k2))

    # For each block, compute the set of (i,j,k) entries it covers
    block_entries = []
    for block in all_blocks:
        i1, i2, j1, j2, k1, k2 = block
        entries = set()
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    entries.add((i, j, k))
        block_entries.append(entries)

    # Greedy + local search for maximum non-overlapping blocks
    # Start with greedy: pick blocks that cover most uncovered entries
    all_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_entries.add((i, j, k))

    # Greedy initialization
    selected = []
    covered = set()
    remaining_blocks = list(range(len(all_blocks)))
    
    # Sort by number of uncovered entries (all initially uncovered)
    # Use a priority: prefer blocks that share fewer entries with other potential blocks
    while remaining_blocks and time.time() < deadline - 50.0:
        best_idx = -1
        best_count = -1
        for idx in remaining_blocks:
            uncov = len(block_entries[idx] - covered)
            if uncov > best_count:
                best_count = uncov
                best_idx = idx
        if best_count <= 0:
            break
        selected.append(best_idx)
        covered.update(block_entries[best_idx])
        remaining_blocks.remove(best_idx)

    # Local search: try to swap out one block for two that cover more
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for pos in range(len(selected)):
            if time.time() > deadline - 30.0:
                break
            # Try removing selected[pos] and adding two new blocks
            saved = selected.pop(pos)
            saved_covered = set(covered)
            covered -= block_entries[saved]
            
            # Find best pair of blocks from remaining
            candidates = [b for b in range(len(all_blocks)) if b not in selected and b != saved]
            best_pair = None
            best_gain = 0
            for ci in range(len(candidates)):
                for cj in range(ci+1, len(candidates)):
                    if time.time() > deadline - 30.0:
                        break
                    e1 = block_entries[candidates[ci]]
                    e2 = block_entries[candidates[cj]]
                    # Must not overlap with current covered
                    if e1 & covered or e2 & covered or (e1 & e2):
                        continue
                    gain = len(e1) + len(e2) - 8  # compared to 1 block covering 8
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (candidates[ci], candidates[cj])
                if time.time() > deadline - 30.0:
                    break
            
            if best_pair and best_gain > 0:
                selected.extend(best_pair)
                covered.update(block_entries[best_pair[0]])
                covered.update(block_entries[best_pair[1]])
                improved = True
            else:
                # Restore
                selected.insert(pos, saved)
                covered = saved_covered

    # Also try single block swaps
    for pos in range(len(selected)):
        if time.time() > deadline - 20.0:
            break
        for alt in range(len(all_blocks)):
            if alt in selected or alt == selected[pos]:
                continue
            if block_entries[alt] & covered:
                continue
            # Check if alt covers entries not currently covered
            if len(block_entries[alt] - covered) > 0:
                # Swap
                saved = selected[pos]
                saved_covered = set(covered)
                covered -= block_entries[saved]
                covered.update(block_entries[alt])
                selected[pos] = alt
                improved = True
                break
        if time.time() > deadline - 20.0:
            break

    num_blocks = len(selected)
    # Total rank: num_blocks * 7 + (105 - num_blocks * 8)
    expected_rank = num_blocks * 7 + (105 - num_blocks * 8)
    expected_rank = max(expected_rank, 105 - num_blocks)  # at minimum, uncovered entries need 1 rank each

    # Now construct the decomposition
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term."""
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
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
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

    # Add selected Strassen blocks
    covered_final = set()
    for idx in selected:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered_final.add((i, j, k))

    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered_final:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Build output matrices
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
>>>>>>> REPLACE

Wait, the local search with pairs is O(n²) per iteration which could be slow. Let me simplify - just do greedy + single swaps, which is faster. Also, the pair search over 630 candidates is 630*629/2 ≈ 198K iterations per swap attempt, which is too slow. Let me remove the pair search and just do greedy + single swaps.

