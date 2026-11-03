import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Schoolbook fallback (rank 60)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen 2x2x2 block decomposition
    # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)*B00       -> C10+=M2, C11-=M2
    # M3 = A00*(B01-B11)       -> C01+=M3, C11+=M3
    # M4 = A11*(B10-B00)       -> C10+=M4, C00+=M4
    # M5 = (A00+A01)*B11       -> C01+=M5, C00-=M5
    # M6 = (A10-A00)*(B00+B01) -> C11+=M6
    # M7 = (A01-A11)*(B10+B11) -> C00+=M7

    # U coefficients: (local_i, local_j) -> coeff
    U_str = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1},      # M7
    ]
    # V coefficients: (local_j, local_k) -> coeff
    V_str = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1},       # M7
    ]
    # W coefficients: (local_k, local_i) -> coeff
    W_str = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,1): 1, (1,1): -1},      # M2
        {(1,0): 1, (1,1): 1},       # M3
        {(0,0): 1, (0,1): 1},       # M4
        {(0,0): -1, (1,0): 1},      # M5
        {(1,1): 1},                 # M6
        {(0,0): 1},                 # M7
    ]

    # Define all possible disjoint Strassen block placements
    # A block covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # We need to find a maximum set of disjoint blocks

    # Generate all valid block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))

    # Greedy: try all blocks, pick those that don't conflict
    # A block (i0,j0,k0) covers:
    #   i indices: {i0, i0+1}
    #   j indices: {j0, j0+1}
    #   k indices: {k0, k0+1}
    # Two blocks conflict if they share any i, j, or k index

    def blocks_conflict(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Check i overlap
        if (i1 == i2) or (i1 == i2 + 1) or (i1 + 1 == i2):
            # i sets overlap
            pass
        else:
            # No i overlap, still need to check j and k
            pass
        # i sets: {i1, i1+1} and {i2, i2+1}
        i_overlap = (i1 == i2) or (i1 == i2 + 1) or (i1 + 1 == i2)
        j_overlap = (j1 == j2) or (j1 == j2 + 1) or (j1 + 1 == j2)
        k_overlap = (k1 == k2) or (k1 == k2 + 1) or (k1 + 1 == k2)
        return i_overlap and j_overlap and k_overlap

    # Actually, for Strassen blocks to be independent, they must not share
    # ANY index in ANY dimension. Let me reconsider.
    # Two blocks (i1,j1,k1) and (i2,j2,k2) are disjoint if:
    # {i1,i1+1} ∩ {i2,i2+1} = ∅ AND {j1,j1+1} ∩ {j2,j2+1} = ∅ AND {k1,k1+1} ∩ {k2,k2+1} = ∅

    def disjoint(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        i_ok = abs(i1 - i2) > 1
        j_ok = abs(j1 - j2) > 1
        k_ok = abs(k1 - k2) > 1
        return i_ok and j_ok and k_ok

    # For n=3, m=4, p=5:
    # i0 can be 0 or 1
    # j0 can be 0, 1, or 2
    # k0 can be 0, 1, 2, or 3

    # Let's enumerate all maximal disjoint sets
    # With n=3, at most 1 block can be used for i-dimension (since i0=0 covers {0,1}, i0=1 covers {1,2}, they overlap)
    # Wait, i0=0 covers {0,1}, i0=1 covers {1,2}. They share index 1, so they conflict.
    # So at most 1 block per i-dimension value... no, at most 1 block total if we need disjoint i sets.
    # Actually with n=3: {0,1} and {1,2} overlap at 1. So we can only have 1 block in i-dimension.
    # But wait - we can have multiple blocks as long as their i-sets are disjoint.
    # With n=3, the only possible i-sets are {0,1} and {1,2}, which overlap. So max 1 block.
    # Hmm, but the inspiration used 4 blocks. Let me re-read.
    # 
    # Looking at inspiration 1: it uses blocks like ([0,1],[0,1],[0,1]), ([0,1],[0,1],[2,3]), etc.
    # These share i={0,1} but have different j and k. So the blocks DON'T need to be fully disjoint!
    # They just need to not double-count any (i,j,k) triple.
    #
    # Actually, for Strassen to work, each 2x2x2 sub-block must be independent.
    # If two blocks share the same i-set but different j,k sets, they're fine because
    # they operate on different (j,k) positions.
    #
    # The key constraint: no (i,j,k) triple should be covered by two different blocks.
    # Block (i0,j0,k0) covers all (i,j,k) where i∈{i0,i0+1}, j∈{j0,j0+1}, k∈{k0,k0+1}.

    def triple_conflict(b1, b2):
        """Check if two blocks share any (i,j,k) triple."""
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        i_overlap = (i1 == i2) or (i1 == i2 + 1) or (i1 + 1 == i2)
        j_overlap = (j1 == j2) or (j1 == j2 + 1) or (j1 + 1 == j2)
        k_overlap = (k1 == k2) or (k1 == k2 + 1) or (k1 + 1 == k2)
        return i_overlap and j_overlap and k_overlap

    # Find maximum set of non-conflicting blocks
    # For n=3, m=4, p=5, let's find good combinations

    # Strategy: enumerate all blocks, find max independent set (greedy + local search)
    # With small search space, we can be more exhaustive

    best_blocks = []
    best_rank = rank_school  # worst case

    # Precompute all blocks
    all_blocks_list = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks_list.append((i0, j0, k0))

    # Greedy approach: try each block as starting point, then greedily add compatible blocks
    for start_block in all_blocks_list:
        selected = [start_block]
        for other in all_blocks_list:
            if other == start_block:
                continue
            # Check compatibility with all selected
            compatible = True
            for sel in selected:
                if triple_conflict(sel, other):
                    compatible = False
                    break
            if compatible:
                selected.append(other)
        
        # Calculate rank for this set of blocks
        # Each block saves 2 (uses 7 instead of 8)
        rank = rank_school - 2 * len(selected)
        if rank < best_rank:
            best_rank = rank
            best_blocks = selected[:]

    # Also try: start with no specific block, greedy from scratch
    selected = []
    for block in all_blocks_list:
        compatible = True
        for sel in selected:
            if triple_conflict(sel, block):
                compatible = False
                break
        if compatible:
            selected.append(block)
    rank = rank_school - 2 * len(selected)
    if rank < best_rank:
        best_rank = rank
        best_blocks = selected[:]

    # Check time
    if time.time() > deadline:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Build the decomposition using best_blocks
    rank_total = best_rank
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    col = 0

    # Add Strassen blocks
    for (i0, j0, k0) in best_blocks:
        for t in range(7):
            # U
            for (li, lj), val in U_str[t].items():
                i = i0 + li
                j = j0 + lj
                U[i * m + j, col] += val
            # V
            for (lj, lk), val in V_str[t].items():
                j = j0 + lj
                k = k0 + lk
                V[j * p + k, col] += val
            # W
            for (lk, li), val in W_str[t].items():
                k = k0 + lk
                i = i0 + li
                W[k * n + i, col] += val
            col += 1

    # Add schoolbook terms for uncovered (i,j,k) triples
    # Mark which triples are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in best_blocks:
        for i in [i0, i0 + 1]:
            for j in [j0, j0 + 1]:
                for k in [k0, k0 + 1]:
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Verify col matches rank_total
    if col != rank_total:
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}

    # Verify correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank_total):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
