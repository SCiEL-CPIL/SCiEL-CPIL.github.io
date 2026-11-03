import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define all possible 2x2x2 block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Build conflict: two blocks conflict if they share any (i,j,k) entry
    # Block (i0,j0,k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # Two blocks conflict if their i-ranges overlap AND j-ranges overlap AND k-ranges overlap
    num_blocks = len(all_blocks)
    
    def blocks_conflict(b1, b2):
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # Check if i-ranges overlap
        if not (i0a <= i0b + 1 and i0b <= i0a + 1):
            return False
        # Check if j-ranges overlap
        if not (j0a <= j0b + 1 and j0b <= j0a + 1):
            return False
        # Check if k-ranges overlap
        if not (k0a <= k0b + 1 and k0b <= k0a + 1):
            return False
        return True
    
    # Greedy: pick blocks that maximize coverage, prefer non-overlapping
    # For 3x5x7, the optimal is 6 blocks: (0,0,0),(0,2,0),(0,0,2),(0,2,2),(0,0,4),(0,2,4)
    # Let's use a simple greedy with most-constrained-first
    
    # Compute degree (number of conflicts) for each block
    degree = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if blocks_conflict(all_blocks[i], all_blocks[j]):
                degree[i] += 1
                degree[j] += 1
    
    # Greedy maximum independent set: repeatedly pick the block with fewest conflicts
    # that doesn't conflict with already picked blocks
    selected = []
    available = set(range(num_blocks))
    
    while available:
        # Pick the block with minimum degree among available
        best_idx = min(available, key=lambda x: degree[x])
        selected.append(best_idx)
        # Remove all blocks that conflict with best_idx
        available.discard(best_idx)
        to_remove = set()
        for other in available:
            if blocks_conflict(all_blocks[best_idx], all_blocks[other]):
                to_remove.add(other)
        available -= to_remove
    
    # Verify: we expect 6 blocks for 3x5x7
    # If greedy didn't find 6, try the known optimal placement
    if len(selected) < 6:
        # Known optimal: (0,0,0), (0,2,0), (0,0,2), (0,2,2), (0,0,4), (0,2,4)
        optimal_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2), (0, 0, 4), (0, 2, 4)]
        # Find indices
        selected = []
        for ob in optimal_blocks:
            for idx, blk in enumerate(all_blocks):
                if blk == ob:
                    selected.append(idx)
                    break
        # Verify no conflicts
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if blocks_conflict(all_blocks[selected[i]], all_blocks[selected[j]]):
                    # Should not happen, but fallback
                    selected = [all_blocks.index(ob) for ob in optimal_blocks]
                    break
    
    # Determine covered entries
    covered = set()
    for idx in selected:
        i0, j0, k0 = all_blocks[idx]
        for i in (i0, i0 + 1):
            for j in (j0, j0 + 1):
                for k in (k0, k0 + 1):
                    covered.add((i, j, k))
    
    # Add Strassen decomposition for each selected block
    def add_strassen_block(i0, j0, k0):
        # M1: U(A00, A11), V(B00, B11), W(C00, C11)
        U = {(i0, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2: U(A10, A11), V(B00), W(C10, -C11)
        U = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1}
        W = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3: U(A00), V(B01, -B11), W(C01, C11)
        U = {(i0, j0): 1}
        V = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        W = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4: U(A11), V(B10, -B00), W(C00, C10)
        U = {(i0+1, j0+1): 1}
        V = {(j0+1, k0): 1, (j0, k0): -1}
        W = {(k0, i0): 1, (k0, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M5: U(A00, A01), V(B11), W(-C00, C01)
        U = {(i0, j0): 1, (i0, j0+1): 1}
        V = {(j0+1, k0+1): 1}
        W = {(k0+1, i0): 1, (k0, i0): -1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M6: U(A10, -A00), V(B00, B01), W(C11)
        U = {(i0+1, j0): 1, (i0, j0): -1}
        V = {(j0, k0): 1, (j0, k0+1): 1}
        W = {(k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M7: U(A01, -A11), V(B10, B11), W(C00)
        U = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        V = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
    
    for idx in selected:
        i0, j0, k0 = all_blocks[idx]
        add_strassen_block(i0, j0, k0)
    
    # Add schoolbook for remaining entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_cols.append({(i, j): 1})
                    V_cols.append({(j, k): 1})
                    W_cols.append({(k, i): 1})
    
    # Convert to dense matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j, c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k, c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i, c] = val
    
    # Verify correctness
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
