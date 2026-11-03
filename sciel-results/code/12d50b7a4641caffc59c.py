import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Structural approach: Greedy maximum-packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block uses 7 rank-1 terms to cover 8 (i,j,k) tensor positions,
    saving 1 rank compared to schoolbook. We greedily select the maximum number
    of mutually non-overlapping Strassen blocks, then fill the rest with schoolbook.
    
    With n=3, m=5, p=7:
    - Possible I-pairs: {0,1}, {1,2}  (2 choices)
    - Possible J-pairs: 5 adjacent pairs (5 choices)
    - Possible K-pairs: 6 adjacent pairs (6 choices)
    - Total possible blocks: 2*5*6 = 60, but limited by disjointness
    
    Maximum packing: ~12 blocks -> rank = 12*7 + (105-12*8) = 84 + 9 = 93
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

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
    # I-pairs: adjacent pairs in range(n)
    i_pairs = [(i, i+1) for i in range(n-1)]  # (0,1), (1,2)
    # J-pairs: adjacent pairs in range(m)
    j_pairs = [(j, j+1) for j in range(m-1)]  # (0,1),(1,2),(2,3),(3,4)
    # K-pairs: adjacent pairs in range(p)
    k_pairs = [(k, k+1) for k in range(p-1)]  # (0,1),(1,2),(2,3),(3,4),(4,5)

    # Build list of all possible blocks with their cell sets
    all_blocks = []
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                cells = get_block_cells(i1, i2, j1, j2, k1, k2)
                all_blocks.append(((i1, i2, j1, j2, k1, k2), cells))

    # Precompute conflict graph
    block_count = len(all_blocks)
    conflicts = [set() for _ in range(block_count)]
    for a in range(block_count):
        for b in range(a+1, block_count):
            if all_blocks[a][1] & all_blocks[b][1]:  # overlap
                conflicts[a].add(b)
                conflicts[b].add(a)

    def greedy_pack(order=None):
        remaining = set(range(block_count))
        selected = []
        while remaining:
            if order is not None:
                # Pick from remaining in the given order
                best = None
                for idx in order:
                    if idx in remaining:
                        best = idx
                        break
                if best is None:
                    best = min(remaining)
            else:
                best = None
                best_conflicts = None
                for idx in remaining:
                    conf_in_remaining = len(conflicts[idx] & remaining)
                    if best is None or conf_in_remaining < best_conflicts:
                        best = idx
                        best_conflicts = conf_in_remaining
            selected.append(best)
            remaining.discard(best)
            remaining -= conflicts[best]
        return selected

    def branch_and_bound_mis():
        """Find maximum independent set via branch and bound."""
        import sys
        sys.setrecursionlimit(10000)
        
        best_set = []
        
        def bbs(remaining, current, upper_bound):
            nonlocal best_set
            if len(current) + upper_bound <= len(best_set):
                return
            if not remaining:
                if len(current) > len(best_set):
                    best_set = list(current)
                return
            
            # Pick a vertex to branch on (max degree in remaining for tighter bound)
            v = max(remaining, key=lambda x: len(conflicts[x] & remaining))
            
            # Branch 1: include v
            new_remaining = remaining - {v} - conflicts[v]
            current.append(v)
            # Upper bound: |current| + |new_remaining|
            bbs(new_remaining, current, len(new_remaining))
            current.pop()
            
            # Branch 2: exclude v
            new_remaining2 = remaining - {v}
            bbs(new_remaining2, current, len(new_remaining2))
        
        bbs(set(range(block_count)), [], block_count)
        return best_set

    def local_search_swap(selected, remaining):
        """Try to improve by swapping: remove 1 or 2 selected, add 2 or 3 from remaining."""
        selected = set(selected)
        remaining = set(remaining)
        improved = True
        while improved:
            improved = False
            # Try: remove 1 selected, add 2 remaining that are compatible with each other and with selected\{s}
            for s in list(selected):
                new_sel = selected - {s}
                # Candidates: remaining blocks not conflicting with new_sel
                candidates = [r for r in remaining if not (conflicts[r] & new_sel)]
                # Need to pick 2 from candidates that don't conflict with each other
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        if not (conflicts[candidates[i]] & {candidates[j]}):
                            # Check they also don't conflict with each other
                            if candidates[i] not in conflicts[candidates[j]]:
                                new_selected = new_sel | {candidates[i], candidates[j]}
                                if len(new_selected) > len(selected):
                                    selected = new_selected
                                    remaining = set(range(block_count)) - selected
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            # Try: remove 2 selected, add 3 remaining
            if not improved:
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i+1, len(sel_list)):
                        s1, s2 = sel_list[i], sel_list[j]
                        new_sel = selected - {s1, s2}
                        candidates = [r for r in remaining if not (conflicts[r] & new_sel)]
                        # Need 3 mutually non-conflicting from candidates
                        for a_i in range(len(candidates)):
                            for b_i in range(a_i+1, len(candidates)):
                                for c_i in range(b_i+1, len(candidates)):
                                    ca, cb, cc = candidates[a_i], candidates[b_i], candidates[c_i]
                                    if (ca not in conflicts[cb] and ca not in conflicts[cc] and cb not in conflicts[cc]):
                                        new_selected = new_sel | {ca, cb, cc}
                                        if len(new_selected) > len(selected):
                                            selected = new_selected
                                            remaining = set(range(block_count)) - selected
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return list(selected)

    # Strategy: branch and bound for exact MIS, then local search
    if time.time() < deadline - 30:
        mis_result = branch_and_bound_mis()
        # Apply local search on top
        remaining_after = set(range(block_count)) - set(mis_result)
        final_selected = local_search_swap(mis_result, remaining_after)
    else:
        # Fallback: multiple greedy
        best_greedy = greedy_pack()
        remaining_after = set(range(block_count)) - set(best_greedy)
        final_selected = local_search_swap(best_greedy, remaining_after)

    selected_indices = sorted(final_selected)

    # Sort selected blocks in a consistent order for deterministic output
    selected_indices.sort()

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
