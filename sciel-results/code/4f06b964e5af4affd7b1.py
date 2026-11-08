import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses branch-and-bound with clique upper bounds to find a maximum packing of disjoint
    2x2x2 Strassen blocks. Each Strassen block reduces the rank by 1 compared to schoolbook.
    Total rank = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 115.0

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
                            all_blocks.append((i1, i2, j1, j2, k1, k2))

    num_blocks_total = len(all_blocks)

    # Convert block indices to bitmasks of covered (i,j,k) positions
    block_bitmasks = []
    for i1, i2, j1, j2, k1, k2 in all_blocks:
        mask = 0
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    idx = i * m * p + j * p + k
                    mask |= (1 << idx)
        block_bitmasks.append(mask)

    # Precompute conflict bitmasks: for each block, a bitmask of which other blocks conflict
    conflict_bitmasks = [0] * num_blocks_total
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflict_bitmasks[i] |= (1 << j)
                conflict_bitmasks[j] |= (1 << i)

    # Precompute degree (number of conflicts) for each block
    degrees = [bin(conflict_bitmasks[i]).count('1') for i in range(num_blocks_total)]

    # Sort blocks by degree descending for better B&B ordering
    block_order = sorted(range(num_blocks_total), key=lambda b: -degrees[b])

    best_num_blocks = 0
    best_packing = []

    def greedy_packing(order=None):
        if order is None:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        
        selected = []
        used_mask = 0
        for b in order:
            if not (block_bitmasks[b] & used_mask):
                selected.append(b)
                used_mask |= block_bitmasks[b]
        return selected

    # Initial greedy solutions with different strategies
    for t in range(100):
        if time.time() > deadline - 50.0:
            break
        
        if t % 4 == 0:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        elif t % 4 == 1:
            order = sorted(range(num_blocks_total), key=lambda b: degrees[b])
            random.shuffle(order)
        elif t % 4 == 2:
            order = sorted(range(num_blocks_total), key=lambda b: -degrees[b])
            random.shuffle(order)
        else:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        
        packing = greedy_packing(order)
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]

    # Compute a greedy upper bound using maximal clique approximation
    def compute_upper_bound(available_blocks):
        """
        Compute an upper bound on the size of an independent set in the induced subgraph
        on available_blocks using a greedy clique-based bound.
        """
        if not available_blocks:
            return 0
        
        # Use a greedy approach: repeatedly pick the vertex with highest degree
        # in the remaining graph and remove it and its neighbors
        remaining = list(available_blocks)
        count = 0
        while remaining:
            # Pick vertex with max degree in remaining
            max_deg = -1
            max_idx = -1
            for idx, v in enumerate(remaining):
                deg = 0
                for w in remaining:
                    if v != w and (conflict_bitmasks[v] >> w) & 1:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    max_idx = idx
            
            count += 1
            # Remove this vertex and all its neighbors
            v = remaining[max_idx]
            to_remove = set()
            to_remove.add(v)
            for w in remaining:
                if (conflict_bitmasks[v] >> w) & 1:
                    to_remove.add(w)
            remaining = [x for x in remaining if x not in to_remove]
        
        return count

    def branch_and_bound(available_mask, current_packing, current_count, depth=0):
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 20.0:
            return
        
        # Get available blocks as a list
        available = [b for b in block_order if (available_mask >> b) & 1]
        
        if not available:
            return
        
        # Upper bound check
        ub = compute_upper_bound(available)
        if current_count + ub <= best_num_blocks:
            return
        
        if current_count > best_num_blocks:
            best_num_blocks = current_count
            best_packing = current_packing[:]
        
        # Score candidates by degree within available set
        scored = []
        for b in available:
            deg = sum(1 for c in available if c != b and (conflict_bitmasks[b] >> c) & 1)
            scored.append((deg, b))
        
        scored.sort(reverse=True)
        
        # Branch on top candidates
        max_branches = min(6, len(scored))
        for _, b in scored[:max_branches]:
            if time.time() > deadline - 20.0:
                return
            
            # Include block b
            new_packing = current_packing + [b]
            new_mask = available_mask & ~(1 << b)
            # Remove all conflicts of b
            for c in available:
                if (conflict_bitmasks[b] >> c) & 1:
                    new_mask &= ~(1 << c)
            
            branch_and_bound(new_mask, new_packing, current_count + 1, depth + 1)
            
            if current_count + 1 > best_num_blocks:
                best_num_blocks = current_count + 1
                best_packing = new_packing[:]
        
        # Exclude the highest-degree candidate
        if scored:
            b = scored[0][1]
            new_mask = available_mask & ~(1 << b)
            branch_and_bound(new_mask, current_packing, current_count, depth + 1)

    # Run B&B from scratch
    full_mask = (1 << num_blocks_total) - 1
    branch_and_bound(full_mask, [], 0)

    # Local search with add moves and 2-for-2 swaps
    for t in range(200):
        if time.time() > deadline - 10.0:
            break
        
        if t % 3 == 0 or not best_packing:
            packing = greedy_packing()
        else:
            packing = best_packing[:]
        
        improved = True
        iterations = 0
        while improved and iterations < 30:
            if time.time() > deadline - 10.0:
                break
            improved = False
            iterations += 1
            
            selected_set = set(packing)
            unselected = [b for b in range(num_blocks_total) if b not in selected_set]
            
            if not unselected:
                break
            
            random.shuffle(unselected)
            
            # Try to add a block
            for new_block in unselected[:40]:
                if time.time() > deadline - 10.0:
                    break
                conflicts = [b for b in packing if (conflict_bitmasks[new_block] >> b) & 1]
                if not conflicts:
                    packing.append(new_block)
                    improved = True
                    if len(packing) > best_num_blocks:
                        best_num_blocks = len(packing)
                        best_packing = packing[:]
                    break
            
            # Try 2-for-2 swaps: remove 2 blocks, add 2 new blocks
            if not improved and len(packing) >= 2:
                swap_found = False
                for i in range(min(15, len(packing))):
                    if swap_found:
                        break
                    for j in range(i + 1, min(15, len(packing))):
                        if time.time() > deadline - 10.0:
                            break
                        r1, r2 = packing[i], packing[j]
                        
                        # Find candidates that conflict with both r1 and r2 but not with others
                        candidates = []
                        for b in unselected[:30]:
                            if (conflict_bitmasks[b] >> r1) & 1 and (conflict_bitmasks[b] >> r2) & 1:
                                # Check no conflict with other packing members
                                ok = True
                                for k, other in enumerate(packing):
                                    if k not in (i, j) and (conflict_bitmasks[b] >> other) & 1:
                                        ok = False
                                        break
                                if ok:
                                    candidates.append(b)
                        
                        if len(candidates) >= 2:
                            # Try pairs of candidates
                            for c1_idx in range(len(candidates)):
                                for c2_idx in range(c1_idx + 1, len(candidates)):
                                    c1, c2 = candidates[c1_idx], candidates[c2_idx]
                                    if not ((conflict_bitmasks[c1] >> c2) & 1):
                                        # Check compatibility with rest
                                        ok = True
                                        for k, other in enumerate(packing):
                                            if k not in (i, j):
                                                if (conflict_bitmasks[c1] >> other) & 1 or (conflict_bitmasks[c2] >> other) & 1:
                                                    ok = False
                                                    break
                                        if ok:
                                            # Perform swap
                                            new_packing = []
                                            for k, b in enumerate(packing):
                                                if k not in (i, j):
                                                    new_packing.append(b)
                                            new_packing.extend([c1, c2])
                                            packing = new_packing
                                            improved = True
                                            swap_found = True
                                            if len(packing) > best_num_blocks:
                                                best_num_blocks = len(packing)
                                                best_packing = packing[:]
                                            break
                                if swap_found:
                                    break
                    if swap_found:
                        break
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
        
        if best_num_blocks >= 14:
            break

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
    for block_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx]
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
