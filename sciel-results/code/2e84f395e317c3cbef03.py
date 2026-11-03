"""Matrix multiplication 3,5,5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """
    Build a rank-minimal tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Returns dict with keys 'U', 'V', 'W' — 2D lists of shape (n*m, rank), (m*p, rank), (p*n, rank).
    The trace convention is W[k*n + i, r].
    Score = rank (lower is better).
    """
    start = time.time()
    deadline = start + 140.0  # safety margin
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Generate all possible 2x2x2 boxes
    # i0 in {0, 1}, j0 in {0, 1, 2, 3}, k0 in {0, 1, 2, 3}
    all_boxes = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_boxes.append((i0, j0, k0))

    def boxes_disjoint(b1, b2):
        # Check if intersection of ranges is empty
        i1_s, i1_e = b1[0], b1[0] + 1
        i2_s, i2_e = b2[0], b2[0] + 1
        j1_s, j1_e = b1[1], b1[1] + 1
        j2_s, j2_e = b2[1], b2[1] + 1
        k1_s, k1_e = b1[2], b1[2] + 1
        k2_s, k2_e = b2[2], b2[2] + 1

        # Overlap in i?
        i_overlap = not (i1_e < i2_s or i2_e < i1_s)
        # Overlap in j?
        j_overlap = not (j1_e < j2_s or j2_e < j1_s)
        # Overlap in k?
        k_overlap = not (k1_e < k2_s or k2_e < k1_s)

        # Disjoint if at least one dimension is disjoint
        return (not i_overlap) or (not j_overlap) or (not k_overlap)

    # Find max set of disjoint boxes
    # Use Backtracking search for max independent set
    # Since N=32, we can try to find size 5.
    
    best_set = []
    
    def backtrack(current_set, index, target_size):
        if len(current_set) == target_size:
            return current_set[:]
        if index == len(all_boxes):
            return None
        
        # Pruning: if remaining boxes + current < target_size, fail
        if len(current_set) + (len(all_boxes) - index) < target_size:
            return None
        
        # Branching bound: check time
        if time.time() > deadline - 10:
            return None

        # Option 1: Take all_boxes[index]
        box = all_boxes[index]
        if all(boxes_disjoint(box, sel) for sel in current_set):
            res = backtrack(current_set + [box], index + 1, target_size)
            if res is not None:
                return res
        
        # Option 2: Skip all_boxes[index]
        res = backtrack(current_set, index + 1, target_size)
        if res is not None:
            return res
        
        return None

    # Try to find 5 disjoint boxes
    found_5 = backtrack([], 0, 5)
    if found_5:
        blocks = found_5
    else:
        # Fallback to 4 (known good)
        found_4 = backtrack([], 0, 4)
        if found_4:
            blocks = found_4
        else:
            # Greedy fallback
            blocks = []
            for b in all_boxes:
                if all(boxes_disjoint(b, sel) for sel in blocks):
                    blocks.append(b)

    num_blocks = len(blocks)
    
    # Count covered triples
    covered = [[[False] * p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0 + di][j0 + dj][k0 + dk] = True
    
    uncovered_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
    total_rank = num_blocks * 7 + uncovered_count

    # Build matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        c = col_offset + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += -1

        c = col_offset + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += -1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        c = col_offset + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] += -1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1

        c = col_offset + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += -1
        W[(k0 + 1) * n + i0][c] += 1

        c = col_offset + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] += -1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        c = col_offset + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += -1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1

    col = 0
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i * m + j][col] += 1
                    V[j * p + k][col] += 1
                    W[k * n + i][col] += 1
                    col += 1

    # Verify
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
