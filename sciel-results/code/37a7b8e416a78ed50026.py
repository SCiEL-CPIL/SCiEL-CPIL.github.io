import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy randomized search to find a large packing of disjoint 2x2x2 Strassen blocks.
    Includes a verification step to ensure correctness, falling back to schoolbook if invalid.
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
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
                            
    best_num_blocks = 0
    best_packing = []
    
    # Greedy Randomized Search
    def greedy_packing(order):
        """Greedy packing with given block index order."""
        current_packing = []
        used_indices = set()
        for idx in order:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        return current_packing

    # Precompute conflict counts for deterministic ordering
    def count_conflicts(block_idx):
        _, block_indices = all_blocks[block_idx]
        return sum(1 for j in range(len(all_blocks)) if j != block_idx and (block_indices & all_blocks[j][1]))

    conflict_counts = [count_conflicts(i) for i in range(len(all_blocks))]
    sorted_by_conflicts = sorted(range(len(all_blocks)), key=lambda i: conflict_counts[i])
    det_packing = greedy_packing(sorted_by_conflicts)
    if len(det_packing) > best_num_blocks:
        best_num_blocks = len(det_packing)
        best_packing = det_packing[:]

    # Local Search Improvement
    def local_search(packing):
        """Try to improve packing by swapping blocks."""
        current_packing = packing[:]
        used_indices = set()
        for bp in current_packing:
            _, inds = all_blocks[[x for x in range(len(all_blocks)) if all_blocks[x][0] == bp][0]]
            used_indices |= inds
            
        improved = True
        while improved:
            improved = False
            # Try to add a block
            for idx in range(len(all_blocks)):
                if idx in [x for x in range(len(all_blocks)) if all_blocks[x][0] in current_packing]:
                    continue
                bp, inds = all_blocks[idx]
                if not (inds & used_indices):
                    current_packing.append(bp)
                    used_indices |= inds
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try to swap: remove one, add two
            for i in range(len(current_packing)):
                remove_bp = current_packing[i]
                remove_idx = [x for x in range(len(all_blocks)) if all_blocks[x][0] == remove_bp][0]
                remove_inds = all_blocks[remove_idx][1]
                
                # Tentatively remove
                temp_packing = current_packing[:i] + current_packing[i+1:]
                temp_used = set()
                for bp in temp_packing:
                    t_idx = [x for x in range(len(all_blocks)) if all_blocks[x][0] == bp][0]
                    temp_used |= all_blocks[t_idx][1]
                    
                # Try to add two blocks
                added = []
                for idx1 in range(len(all_blocks)):
                    if idx1 == remove_idx: continue
                    if idx1 in [x for x in range(len(all_blocks)) if all_blocks[x][0] in temp_packing]: continue
                    bp1, inds1 = all_blocks[idx1]
                    if not (inds1 & temp_used):
                        added.append(idx1)
                        temp_used |= inds1
                        # Try to add a second one
                        for idx2 in range(len(all_blocks)):
                            if idx2 == remove_idx or idx2 == idx1: continue
                            if idx2 in [x for x in range(len(all_blocks)) if all_blocks[x][0] in temp_packing]: continue
                            bp2, inds2 = all_blocks[idx2]
                            if not (inds2 & temp_used):
                                # Success: Swap 1 for 2
                                new_packing = temp_packing + [bp1, bp2]
                                if len(new_packing) > len(current_packing):
                                    return new_packing
                                break
                        if len(added) > 0:
                            break
                if added:
                    # Revert if no improvement found in inner loop, or continue search
                    pass

        return current_packing

    # Randomized trials with local search
    num_trials = 100
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break
            
        available_indices = list(range(len(all_blocks)))
        random.shuffle(available_indices)
        current_packing = greedy_packing(available_indices)
        
        # Apply local search
        improved_packing = local_search(current_packing)
        
        if len(improved_packing) > best_num_blocks:
            best_num_blocks = len(improved_packing)
            best_packing = improved_packing[:]
            
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
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(np.dot(U_np[u_idx], V_np[v_idx]), W_np[w_idx])
                        if not np.isclose(val, 1.0, atol=1e-9):
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
