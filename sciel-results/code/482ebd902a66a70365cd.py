import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses backtracking with constraint propagation to find a maximum packing of disjoint
    2x2x2 Strassen blocks, then fills remaining triples with schoolbook terms.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | frozenset([(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks_total = len(all_blocks)
    
    # Precompute conflict matrix: block_a conflicts with block_b if they share any triple
    # For efficiency, use bitsets or just set intersection
    # With 630 blocks, a 630x630 boolean matrix is ~400KB, manageable
    
    # Use a more efficient approach: for each block, store the set of triples as a frozenset
    # and use set intersection for conflict checking
    
    # Backtracking search for maximum packing
    best_packing = []
    best_size = 0
    
    # Sort blocks by a heuristic: blocks that conflict with more other blocks first
    # (most constrained first)
    conflict_counts = [0] * num_blocks_total
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict_counts[i] += 1
                conflict_counts[j] += 1
    
    # Order blocks: most constrained first
    block_order = sorted(range(num_blocks_total), key=lambda x: -conflict_counts[x])
    
    # Backtracking with pruning
    used_indices = set()
    current_packing = []
    
    def backtrack(idx, remaining_blocks):
        nonlocal best_packing, best_size
        
        if time.time() > deadline - 5.0:
            return
        
        # Pruning: even if we add all remaining blocks, can we beat best?
        if len(current_packing) + len(remaining_blocks) <= best_size:
            return
        
        if not remaining_blocks:
            if len(current_packing) > best_size:
                best_size = len(current_packing)
                best_packing = current_packing[:]
            return
        
        # Try skipping the next block
        backtrack(idx + 1, remaining_blocks[1:])
        
        # Try adding the next block if it doesn't conflict
        block_idx = remaining_blocks[0]
        block_params, block_indices = all_blocks[block_idx]
        if not (block_indices & used_indices):
            current_packing.append(block_params)
            used_indices |= block_indices
            backtrack(idx + 1, remaining_blocks[1:])
            current_packing.pop()
            used_indices -= block_indices
    
    # The above naive backtracking will be too slow with 630 blocks.
    # Let's use a greedy + local search approach instead, with better heuristics.
    
    # Improved greedy with multiple random restarts and local search
    best_num_blocks = 0
    best_packing = []
    
    # Precompute for each block the set of triples it covers (as frozenset for fast intersection)
    block_triplets = [b[1] for b in all_blocks]
    
    num_trials = 500
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break
        
        random.seed(t + 12345)  # Reproducible with offset
        available_blocks = list(range(num_blocks_total))
        
        # Sort by a heuristic: prefer blocks that cover more "rare" triples
        # First, count how many blocks cover each triple
        triple_count = {}
        for bidx in range(num_blocks_total):
            for tri in block_triplets[bidx]:
                triple_count[tri] = triple_count.get(tri, 0) + 1
        
        # Score each block by sum of inverse frequencies of its triples
        block_scores = []
        for bidx in range(num_blocks_total):
            score = sum(1.0 / triple_count[tri] for tri in block_triplets[bidx])
            block_scores.append(score)
        
        random.shuffle(available_blocks)
        
        current_packing = []
        used_indices = set()
        
        for block_idx in available_blocks:
            if time.time() > deadline - 15.0:
                break
            block_params, block_indices = all_blocks[block_idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        
        # Local search: 2-opt style - remove one block, try to add up to 2 new blocks
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            current_set = set(current_packing)
            used = set()
            for bp in current_packing:
                i1, i2, j1, j2, k1, k2 = bp
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            used.add((i, j, k))
            
            best_gain = 0
            best_move = None
            
            for remove_idx, remove_bp in enumerate(current_packing):
                if time.time() > deadline - 15.0:
                    break
                ri1, ri2, rj1, rj2, rk1, rk2 = remove_bp
                removed_indices = frozenset()
                for i in (ri1, ri2):
                    for j in (rj1, rj2):
                        for k in (rk1, rk2):
                            removed_indices = removed_indices | frozenset([(i, j, k)])
                
                new_used = used - removed_indices
                
                # Try to greedily add non-conflicting blocks
                new_packing = current_packing[:remove_idx] + current_packing[remove_idx+1:]
                new_used_set = set(new_used)
                
                added = 0
                for block_idx in available_blocks:
                    if time.time() > deadline - 15.0:
                        break
                    block_params, block_indices = all_blocks[block_idx]
                    if block_params in current_set or block_params == remove_bp:
                        continue
                    if not (block_indices & new_used_set):
                        new_packing.append(block_params)
                        new_used_set |= block_indices
                        added += 1
                        if added >= 2:
                            break
                
                gain = added - 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (remove_idx, new_packing)
            
            if best_move is not None and best_gain > 0:
                remove_idx, new_packing = best_move
                current_packing = new_packing
                improved = True
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
    
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
        # A is 2x2: rows i1,i2; cols j1,j2
        # B is 2x2: rows j1,j2; cols k1,k2
        # C is 2x2: rows k1,k2; cols i1,i2
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
        
        # Standard Strassen for C = A*B where:
        # A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
        # C = [[c11, c12],[c21, c22]]
        #
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
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

    # Verification Step
    if rank > 0:
        try:
            U_np = np.array(U, dtype=np.int64)
            V_np = np.array(V, dtype=np.int64)
            W_np = np.array(W, dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # Compute sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                        prod = U_np[u_idx] * V_np[v_idx]
                        val = np.dot(prod, W_np[w_idx])
                        
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if not valid:
                # Fall back to schoolbook
                U_cols = []
                V_cols = []
                W_cols = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
                            
                rank = len(U_cols)
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
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
