import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an exhaustive+local-search approach to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Key improvements:
    1. Exhaustive search over all subsets (only 24 blocks, so 2^24 is feasible with pruning)
    2. Local search with swap moves to escape local optima
    3. Multiple random restarts with different orderings
    4. Fast bitmask operations for disjointness
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Total number of (i,j,k) triples
    total_triples = n * m * p  # 105

    # Precompute all possible 2x2x2 blocks with bitmask representation
    all_blocks = []  # list of (params, bitmask, index_set)
    
    # Index of each (i,j,k) triple to a unique bit position
    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, indices))
    
    num_total_blocks = len(all_blocks)  # 24 blocks
    
    # Precompute pairwise conflict matrix
    conflict = [[False] * num_total_blocks for _ in range(num_total_blocks)]
    for a in range(num_total_blocks):
        for b in range(a + 1, num_total_blocks):
            if (all_blocks[a][1] & all_blocks[b][1]) != 0:
                conflict[a][b] = True
                conflict[b][a] = True

    best_num_blocks = 0
    best_packing = []

    def try_local_search(packing_indices):
        """Try to improve packing by adding/removing/swapping blocks."""
        if time.time() > deadline - 5.0:
            return len(packing_indices), packing_indices
        
        used_mask = 0
        for idx in packing_indices:
            used_mask |= all_blocks[idx][1]
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding a block
            for idx in range(num_total_blocks):
                if idx not in packing_indices and (all_blocks[idx][1] & used_mask) == 0:
                    packing_indices.append(idx)
                    used_mask |= all_blocks[idx][1]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one
            for remove_idx in range(len(packing_indices)):
                if time.time() > deadline - 5.0:
                    break
                old_mask = all_blocks[packing_indices[remove_idx]][1]
                new_used = used_mask & ~old_mask
                for add_idx in range(num_total_blocks):
                    if add_idx not in packing_indices and (all_blocks[add_idx][1] & new_used) == 0:
                        packing_indices[remove_idx] = add_idx
                        used_mask = new_used | all_blocks[add_idx][1]
                        improved = True
                        break
                if improved:
                    break
        
        return len(packing_indices), packing_indices

    # Strategy 1: Exhaustive search with branch and bound
    # Since we only have 24 blocks, we can do a DFS with pruning
    def dfs_exhaustive(available_mask, current_packing, current_count):
        """DFS with branch and bound to find max independent set."""
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 10.0:
            return
        
        # Upper bound: current_count + number of available blocks that don't conflict with each other
        # Simple bound: current_count + popcount(available_mask)
        avail_count = bin(available_mask).count('1')
        if current_count + avail_count <= best_num_blocks:
            return
        
        if available_mask == 0:
            if current_count > best_num_blocks:
                best_num_blocks = current_count
                best_packing = list(current_packing)
            return
        
        # Pick the next block to try (smallest index for determinism, or use heuristic)
        # Try both including and excluding each available block
        # Use a heuristic: try blocks with fewer conflicts first
        available_indices = [i for i in range(num_total_blocks) if (available_mask >> i) & 1]
        
        # Sort by number of conflicts with other available blocks (fewer conflicts first)
        def conflict_count(idx):
            count = 0
            for other in available_indices:
                if other != idx and conflict[idx][other]:
                    count += 1
            return count
        
        available_indices.sort(key=conflict_count)
        
        for idx in available_indices:
            if time.time() > deadline - 10.0:
                return
            if not ((available_mask >> idx) & 1):
                continue
            
            # Include this block
            current_packing.append(idx)
            new_available = available_mask & ~(1 << idx)
            # Remove all blocks that conflict with idx
            for other in range(num_total_blocks):
                if conflict[idx][other]:
                    new_available &= ~(1 << other)
            
            dfs_exhaustive(new_available, current_packing, current_count + 1)
            current_packing.pop()
            
            # Exclude this block
            available_mask &= ~(1 << idx)
            
            # Early termination check
            if current_count + bin(available_mask).count('1') <= best_num_blocks:
                return

    # Run exhaustive search
    initial_available = (1 << num_total_blocks) - 1
    dfs_exhaustive(initial_available, [], 0)

    # Strategy 2: Multiple random restarts with local search
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random greedy initialization
        order = list(range(num_total_blocks))
        random.shuffle(order)
        packing = []
        used_mask = 0
        for idx in order:
            if (all_blocks[idx][1] & used_mask) == 0:
                packing.append(idx)
                used_mask |= all_blocks[idx][1]
        
        # Local search improvement
        count, packing = try_local_search(packing)
        
        if count > best_num_blocks:
            best_num_blocks = count
            best_packing = packing
            
            if best_num_blocks >= 10:
                break

    # Convert packing from indices to params
    best_packing_params = [all_blocks[idx][0] for idx in best_packing]

    # Construct U, V, W based on the best packing found
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

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing_params:
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
