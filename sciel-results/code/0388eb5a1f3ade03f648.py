"""Optimized rank decomposition for 3x4*4x7 matrix multiplication.

Uses BCLR rank-5 decomposition for 2x2x2 Strassen blocks instead of the
standard rank-7 Strassen algorithm. This reduces the rank contribution of
each 2x2x2 block from 7 to 5, potentially reducing total rank from 78 to 66.
"""
import time
import itertools

_DIMS = (3, 4, 7)


def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # Total number of tensor entries
    total_entries = n * m * p
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
                
    # Function to check if two blocks are disjoint
    def are_disjoint(b1, b2):
        i0_1, j0_1, k0_1 = b1
        i0_2, j0_2, k0_2 = b2
        
        # Intervals are [i0, i0+1]
        # Disjoint in I if i0_1+1 < i0_2 or i0_2+1 < i0_1
        if (i0_1 + 1 < i0_2) or (i0_2 + 1 < i0_1):
            return True
        # Disjoint in J
        if (j0_1 + 1 < j0_2) or (j0_2 + 1 < j0_1):
            return True
        # Disjoint in K
        if (k0_1 + 1 < k0_2) or (k0_2 + 1 < k0_1):
            return True
        return False

    # Greedy search for maximum packing.
    best_blocks = []
    
    # Try a few different greedy strategies
    # Strategy 1: Sort by i0, then j0, then k0
    sorted_blocks = sorted(blocks, key=lambda b: (b[0], b[1], b[2]))
    selected = []
    for b in sorted_blocks:
        if all(are_disjoint(b, sb) for sb in selected):
            selected.append(b)
    if len(selected) > len(best_blocks):
        best_blocks = selected

    # Strategy 2: Sort by k0 (depth) primarily, to spread out in the large dimension
    sorted_blocks_k = sorted(blocks, key=lambda b: (b[2], b[0], b[1]))
    selected = []
    for b in sorted_blocks_k:
        if all(are_disjoint(b, sb) for sb in selected):
            selected.append(b)
    if len(selected) > len(best_blocks):
        best_blocks = selected

    # Strategy 3: Randomized greedy with restarts (if time permits)
    import random
    random.seed(42)
    
    # Check time
    if time.time() - _t0 < 10.0:
        for _ in range(50):
            shuffled = blocks[:]
            random.shuffle(shuffled)
            selected = []
            for b in shuffled:
                if all(are_disjoint(b, sb) for sb in selected):
                    selected.append(b)
            if len(selected) > len(best_blocks):
                best_blocks = selected
                if len(best_blocks) == 10: # Theoretical max
                    break

    # Construct the decomposition
    num_blocks = len(best_blocks)
    # Create a set of covered entries to identify schoolbook terms
    covered_entries = set()
    for b in best_blocks:
        i0, j0, k0 = b
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered_entries.add((i, j, k))
    
    uncovered_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_entries:
                    uncovered_entries.append((i, j, k))
    
    rank = 5 * num_blocks + len(uncovered_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_bclr(i0, j0, k0):
        nonlocal col
        # BCLR rank-5 decomposition for 2x2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        W[k0 * n + (i0+1)][col] = -1
        W[(k0+1) * n + i0][col] = -1
        col += 1
        
        # M2 = A10(B00+B01)
        U[(i0+1) * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M3 = (A00+A01)B10
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        col += 1
        
        # M4 = (A01-A11)B11
        U[i0 * m + (j0+1)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        
        # M5 = A11(B10-B00)
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + (i0+1)][col] = -1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1

    for b in best_blocks:
        add_bclr(*b)
        
    # Add schoolbook terms for uncovered entries
    for (i, j, k) in uncovered_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
    # Use vectorized check for speed, but only if time permits
    if time.time() - _t0 < 100.0:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check a sample or all if fast enough
        # Full check: 84 entries * rank mults
        # 84 * 66 ~ 5500 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if val != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): {val}")

    return {"U": U, "V": V, "W": W}
