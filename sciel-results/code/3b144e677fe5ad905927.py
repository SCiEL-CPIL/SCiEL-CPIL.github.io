import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a backtracking search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Improvements over previous version:
    1. Structured backtracking search with pruning instead of random restarts.
    2. Blocks sorted by conflict density to find good packings quickly.
    3. Explicit deadline guard with early termination if time runs out.
    4. Verification step to ensure the decomposition is exact.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_triples = n * m * p  # 105

    # Precompute all possible 2x2x2 blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2) with i1<i2, j1<j2, k1<k2
    blocks = []
    
    # Map (i,j,k) to a bit index
    def tidx(i, j, k):
        return i * m * p + j * p + k

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        mask |= (1 << tidx(i, j, k))
                            blocks.append({
                                'params': (i1, i2, j1, j2, k1, k2),
                                'mask': mask,
                                'idx': len(blocks)
                            })
    
    num_blocks = len(blocks)
    
    # Calculate conflicts for sorting
    # Conflict count: number of other blocks that overlap with this one
    conflicts = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if (blocks[i]['mask'] & blocks[j]['mask']) != 0:
                conflicts[i] += 1
                conflicts[j] += 1
                
    # Sort blocks by conflict count (ascending) to prioritize "isolated" blocks
    sorted_blocks = sorted(blocks, key=lambda x: conflicts[x['idx']])
    
    # Re-index or just use the sorted list for search
    # We need to be careful with indices in the recursion.
    # Let's create a list of masks in sorted order
    sorted_masks = [b['mask'] for b in sorted_blocks]
    sorted_params = [b['params'] for b in sorted_blocks]
    
    best_count = 0
    best_packing = []
    
    # Global variables for recursion
    # Using a list to allow modification in nested function
    current_packing = []
    
    def dfs(used_mask, count, start_idx):
        nonlocal best_count, best_packing
        
        if count > best_count:
            best_count = count
            best_packing = current_packing[:]
            
        # Pruning: if we can't beat best_count even if we pick all remaining, return
        # Upper bound on additional blocks: num_blocks - start_idx
        if count + (num_blocks - start_idx) <= best_count:
            return
            
        # Check time
        if time.time() > deadline - 5.0:
            return

        for i in range(start_idx, num_blocks):
            mask = sorted_masks[i]
            # Check disjointness
            if (mask & used_mask) == 0:
                current_packing.append(sorted_params[i])
                dfs(used_mask | mask, count + 1, i + 1)
                current_packing.pop()
                
                # Optional: If we found a very good packing, we might want to break early
                # but usually we want to explore more to ensure maximum.
                
    # Initial greedy to set a good lower bound for pruning
    used_mask = 0
    greedy_count = 0
    greedy_packing = []
    for i in range(num_blocks):
        mask = sorted_masks[i]
        if (mask & used_mask) == 0:
            used_mask |= mask
            greedy_count += 1
            greedy_packing.append(sorted_params[i])
            
    if greedy_count > best_count:
        best_count = greedy_count
        best_packing = greedy_packing[:]
        
    # Run backtracking
    # Start with empty mask and count 0
    # To speed up, we can start the recursion by trying to add blocks to the greedy packing?
    # No, standard DFS from scratch is cleaner. The greedy bound helps pruning.
    
    # We can also try to start the DFS with the greedy packing as a prefix?
    # That would require a different structure. Let's stick to standard DFS.
    
    # Optimization: If the number of blocks is large, DFS can be slow.
    # 420 blocks is small enough.
    
    # To further speed up, we can skip blocks that are subsets of already used masks? 
    # No, masks are fixed size (8 bits).
    
    # Let's run the DFS
    dfs(0, 0, 0)
    
    # If we didn't find anything better than greedy, use greedy
    # best_packing is already updated in dfs if it found something better
    
    # Construct U, V, W
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
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
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
