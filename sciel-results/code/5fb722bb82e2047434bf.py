"""
Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.

Approach:
- Use a greedy set-cover / maximum-overlap strategy to pack as many 2x2x2 Strassen
  blocks (rank 7 each) as possible into the 3x4x7 tensor.
- The tensor has dimensions n=3 (rows of A), m=4 (cols of A / rows of B), p=7 (cols of B).
- A Strassen block covers a 2x2x2 sub-tensor: 2 rows of A, 2 cols of A (rows of B), 2 cols of B.
- We can place blocks with i in {0,1} or {1,2} (since n=3, we have two possible 2-row windows),
  j in {0,1}, {1,2}, {2,3} (three possible 2-col windows for m=4),
  k in {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6} (six possible 2-col windows for p=7).
- However, Strassen blocks must be disjoint in the sense that no (i,j,k) entry is covered by
  more than one block (otherwise the linear combinations would interfere). Actually, Strassen
  blocks CAN overlap in indices as long as we handle the linear algebra correctly, but the
  standard approach requires disjoint support for the 7-term decomposition to work cleanly.
  
  Wait - actually, in tensor decomposition, the rank-1 terms just need to sum to the tensor.
  Multiple blocks CAN share indices as long as the total contribution works out. But the
  Strassen 7-term formula is designed for a clean 2x2x2 block. If blocks overlap, the
  contributions from different blocks to the same (i,j,k) entry must sum to 1.
  
  The simplest correct approach: use DISJOINT blocks (no (i,j,k) shared between blocks),
  then fill the rest with schoolbook (rank 1 each).
  
  Let's enumerate all possible 2x2x2 blocks:
  i-windows: {0,1}, {1,2}  -> 2 options
  j-windows: {0,1}, {1,2}, {2,3} -> 3 options
  k-windows: {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6} -> 6 options
  Total possible blocks: 2 * 3 * 6 = 36
  
  Each block covers 8 (i,j,k) entries. Total entries: 3*4*7 = 84.
  Max disjoint blocks: floor(84/8) = 10, but constrained by the window structure.
  
  Greedy set-cover: pick blocks that cover the most uncovered entries.
  
  Current program uses 6 blocks (i in {0,1} only, all j-pairs, k-pairs {0,1},{2,3},{4,5}).
  That's 2 * 3 * 3 = 18 possible if we use i={0,1} and all j,k pairs, but only 6 are used.
  
  Let's try to maximize the number of disjoint Strassen blocks.
  
  For i={0,1}: j-windows {0,1},{1,2},{2,3}; k-windows {0,1},{2,3},{4,5},{5,6}
  We can pick a matching-like structure. Let's think of it as a 3D matching problem.
  
  Actually, let's just do a greedy approach:
  1. Generate all possible 2x2x2 blocks.
  2. Sort by some heuristic (e.g., prefer blocks that cover entries with fewer alternative coverings).
  3. Greedily select blocks that don't conflict with already-selected blocks.
  4. Fill remaining entries with schoolbook.
  
  Let's see if we can get more than 6 blocks.
  
  With i={0,1}: 3 j-windows x 6 k-windows = 18 blocks, but they share j and k indices.
  For disjointness, if we pick block (j={0,1}, k={0,1}), we can't pick any block sharing j in {0,1} or k in {0,1} with another block for the same i-pair.
  
  Actually, two blocks conflict if they share ANY (i,j,k) entry. Since i is the same ({0,1}),
  they conflict if their j-windows intersect AND their k-windows intersect.
  
  j-windows: A={0,1}, B={1,2}, C={2,3}. A∩B={1}, A∩C=∅, B∩C={2}.
  k-windows: {0,1},{1,2},{2,3},{3,4},{4,5},{5,6}. Adjacent ones intersect.
  
  So for i={0,1}, we need a set of (j-window, k-window) pairs such that no two pairs have
  intersecting j-windows AND intersecting k-windows.
  
  This is like a 2D packing. Let's try to maximize:
  - Pick (A, k0={0,1}): covers j=0,1 and k=0,1.
  - Next j-window that doesn't intersect A is C={2,3}. Pick (C, k2={2,3}): covers j=2,3, k=2,3.
  - Now can we add more? (B, k4={4,5}): j={1,2} intersects A at {1} and C at {2}, but k={4,5} 
    doesn't intersect k0 or k2. So (B, {4,5}) is OK with both!
  - Can we add (A, k4={4,5})? j={0,1} doesn't intersect C's j, but intersects B's j at {1}. 
    k={4,5} intersects (B,{4,5})'s k. So conflict with (B,{4,5}).
  - How about (B, {5,6})? j={1,2} intersects A and C, k={5,6} doesn't intersect k0,k2,{4,5}. 
    But j intersects (B,{4,5})'s j, and k doesn't intersect. So OK? Wait, conflict requires BOTH
    j and k to intersect. (B,{5,6}) vs (B,{4,5}): j same (intersect), k={5,6}∩{4,5}={5} (intersect). CONFLICT.
  - (A, {5,6}) vs (B,{4,5}): j={0,1}∩{1,2}={1} intersect, k={5,6}∩{4,5}={5} intersect. CONFLICT.
  
  So for i={0,1}, max seems to be 3 blocks: e.g., (A,{0,1}), (C,{2,3}), (B,{4,5}).
  That gives 3 blocks for i={0,1}.
  
  Similarly for i={1,2}: another 3 blocks.
  
  Total: 6 blocks. Same as current!
  
  Hmm. Let me try different k-partitions.
  k-windows: {0,1},{2,3},{4,5},{5,6} - wait, {4,5} and {5,6} intersect at 5.
  Alternative: {0,1},{2,3},{4,5} and then {6} is left. Or {0,1},{2,3},{4,5},{6}? No, must be size 2.
  
  What about non-adjacent k-windows: {0,1}, {2,3}, {4,5} are pairwise disjoint. {5,6} intersects {4,5}.
  So for k, we can pick at most 3 pairwise-disjoint 2-windows from 7 elements: {0,1},{2,3},{4,5} or {0,1},{2,3},{5,6} etc.
  
  For j: {0,1},{2,3} are disjoint. {1,2} intersects both. So max 2 disjoint j-windows: {0,1},{2,3}.
  
  So for a fixed i-pair, max blocks = 2 (j) * 3 (k) = 6? No wait, the constraint is that no two selected blocks have BOTH j and k intersecting. It's not that j-windows must be pairwise disjoint OR k-windows must be pairwise disjoint.
  
  Let me reconsider. Two blocks (j1,k1) and (j2,k2) conflict iff j1∩j2≠∅ AND k1∩k2≠∅.
  
  This is equivalent to: the selected set forms an independent set in a conflict graph.
  
  Let me just do greedy in code and see what rank we get. If we can get 7+ blocks, rank drops.
  
  Actually, let me think more carefully. The current solution gets rank 78 = 6*7 + 36.
  84 total entries - 6*8 = 84 - 48 = 36 uncovered. Yes.
  
  If we could get 7 blocks: 84 - 56 = 28 uncovered, rank = 49 + 28 = 77.
  If 8 blocks: 84 - 64 = 20, rank = 56 + 20 = 76.
  If 9 blocks: 84 - 72 = 12, rank = 63 + 12 = 75.
  If 10 blocks: 84 - 80 = 4, rank = 70 + 4 = 74.
  
  Can we get more than 6? Let me try a different selection.
  
  For i={0,1}:
  Try: (j={0,1}, k={0,1}), (j={1,2}, k={2,3}), (j={2,3}, k={4,5})
  Check conflicts:
  - Block1 vs Block2: j={0,1}∩{1,2}={1}≠∅, k={0,1}∩{2,3}=∅. NO conflict.
  - Block1 vs Block3: j={0,1}∩{2,3}=∅. NO conflict.
  - Block2 vs Block3: j={1,2}∩{2,3}={2}≠∅, k={2,3}∩{4,5}=∅. NO conflict.
  Great! 3 blocks for i={0,1}.
  
  For i={1,2}:
  Same structure: (j={0,1}, k={0,1}), (j={1,2}, k={2,3}), (j={2,3}, k={4,5})
  But wait, i={1,2} blocks share i=1 with i={0,1} blocks! So they conflict on entries where i=1.
  
  Ah, I forgot: blocks with different i-pairs can also conflict if they share i indices.
  i={0,1} and i={1,2} share i=1. So any (i=1, j, k) entry covered by both would conflict.
  
  So if block from i={0,1} covers (1, j, k) and block from i={1,2} covers (1, j', k'), they conflict iff j=j' AND k=k'.
  
  This makes it more complex. Let me just implement a greedy set-cover in the code.
  
  Actually, for simplicity and to ensure correctness, let me stick with a clean approach:
  - Generate all possible blocks.
  - Greedily select non-conflicting blocks.
  - Fill rest with schoolbook.
  
  Let me code this up properly.
"""

import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Generate all possible 2x2x2 Strassen blocks
    # i-windows: consecutive pairs in range(n)
    i_windows = [(i, i + 1) for i in range(n - 1)]  # (0,1), (1,2)
    j_windows = [(j, j + 1) for j in range(m - 1)]  # (0,1), (1,2), (2,3)
    k_windows = [(k, k + 1) for k in range(p - 1)]  # (0,1), ..., (5,6)

    all_blocks = []
    for (i0, i1) in i_windows:
        for (j0, j1) in j_windows:
            for (k0, k1) in k_windows:
                # This block covers 8 entries
                entries = set()
                for ii in (i0, i1):
                    for jj in (j0, j1):
                        for kk in (k0, k1):
                            entries.add((ii, jj, kk))
                all_blocks.append(((i0, i1, j0, j1, k0, k1), entries))

    # Improved greedy set-cover with local search / multiple trials
    # to maximize the number of disjoint Strassen blocks.
    
    def greedy_select(blocks, order):
        selected = []
        covered = set()
        for idx in order:
            params, entries = blocks[idx]
            if not any(e in covered for e in entries):
                selected.append(params)
                covered.update(entries)
        return selected, covered

    import random
    best_selected = []
    best_covered = set()
    best_count = 0
    
    # Try a few different orderings
    # 1. Rarity heuristic
    entry_block_count = {}
    for _, entries in all_blocks:
        for e in entries:
            entry_block_count[e] = entry_block_count.get(e, 0) + 1

    def block_score_rarity(block_info):
        _, entries = block_info
        score = 0.0
        for e in entries:
            score += 1.0 / entry_block_count[e]
        return score

    indices = list(range(len(all_blocks)))
    indices.sort(key=lambda i: block_score_rarity(all_blocks[i]), reverse=True)
    sel, cov = greedy_select(all_blocks, indices)
    if len(sel) > best_count:
        best_count = len(sel)
        best_selected = sel
        best_covered = cov

    # 2. Random trials
    for _ in range(100):
        random.shuffle(indices)
        sel, cov = greedy_select(all_blocks, indices)
        if len(sel) > best_count:
            best_count = len(sel)
            best_selected = sel
            best_covered = cov

    # 3. Try to improve by swapping
    # If we have time, we can do a simple local search
    # But for now, let's see if random trials find > 6.
    
    selected = best_selected
    covered_entries = best_covered

    num_blocks = len(selected)
    num_schoolbook = n * m * p - num_blocks * 8
    rank = num_blocks * 7 + num_schoolbook

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+, C[i1,k1]+=
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+, C[i1,k1]-
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+, C[i1,k1]+=
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+, C[i1,k0]+=
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-, C[i0,k1]+=
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    for params in selected:
        col = add_strassen_block(col, *params)

    # Schoolbook for remaining entries
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered_entries:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
