import time
import numpy as np

_DIMS = (3, 4, 7)

def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # Total entries
    total_entries = n * m * p
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
                
    num_blocks = len(blocks)
    
    # Conflict matrix: conflict[i][j] is True if blocks i and j are NOT disjoint
    # Disjoint if intervals don't overlap in at least one dimension.
    # Overlap in dim X if |i0_1 - i0_2| < 2.
    # So conflict if |i0_1 - i0_2| < 2 AND |j0_1 - j0_2| < 2 AND |k0_1 - k0_2| < 2.
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        i0, j0, k0 = blocks[i]
        for j in range(i + 1, num_blocks):
            i0j, j0j, k0j = blocks[j]
            if (abs(i0 - i0j) < 2 and abs(j0 - j0j) < 2 and abs(k0 - k0j) < 2):
                conflict[i][j] = True
                conflict[j][i] = True

    # Find maximum independent set (max disjoint blocks)
    # Use branch and bound
    # Order nodes by degree descending to prune earlier
    degrees = [sum(conflict[i]) for i in range(num_blocks)]
    order = sorted(range(num_blocks), key=lambda x: -degrees[x])
    
    best_set = []
    current_set = []
    
    # Map ordered indices to original indices
    # We will recurse on the ordered list
    # To speed up, we can use a bitset or just list. 36 nodes is small.
    
    def is_disjoint_from_current(idx):
        for c_idx in current_set:
            if conflict[idx][c_idx]:
                return False
        return True

    def backtrack(remaining_indices, current_count):
        nonlocal best_set
        # Pruning
        if current_count + len(remaining_indices) <= len(best_set):
            return
        
        if not remaining_indices:
            if current_count > len(best_set):
                best_set = current_set[:]
            return

        # Pick the first available node
        idx = remaining_indices[0]
        rest = remaining_indices[1:]
        
        # Option 1: Include idx
        if is_disjoint_from_current(idx):
            current_set.append(idx)
            # Filter rest to only those disjoint from idx
            new_rest = [r for r in rest if not conflict[idx][r]]
            backtrack(new_rest, current_count + 1)
            current_set.pop()
            
        # Option 2: Exclude idx
        backtrack(rest, current_count)

    # Initial call
    backtrack(order, 0)
    
    # best_set contains indices into the 'blocks' list
    num_strassen = len(best_set)
    
    # Determine covered entries
    covered = set()
    for b_idx in best_set:
        i0, j0, k0 = blocks[b_idx]
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered.add((i, j, k))
                    
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.append((i, j, k))
                    
    rank = 7 * num_strassen + len(uncovered)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M2
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        # M5
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        # M6
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M7
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b_idx in best_set:
        add_strassen(*blocks[b_idx])
        
    # Add schoolbook terms
    for (i, j, k) in uncovered:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
    if time.time() < _t0 + 140:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
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
