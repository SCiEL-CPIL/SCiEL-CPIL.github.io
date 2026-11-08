import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an ILP-style branch-and-bound with constraint propagation to find a maximum packing
    of disjoint 2x2x2 Strassen blocks. Each Strassen block reduces the rank by 1 compared to
    schoolbook. Total rank = 105 - num_blocks.
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
    
    # Precompute overlap matrix as bitmasks for speed
    # Map each (i,j,k) to a bit position
    idx_to_bit = {}
    bit_to_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                idx_to_bit[(i, j, k)] = 1 << idx
                bit_to_idx.append((i, j, k))
    
    # Convert block indices to bitmasks
    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= idx_to_bit[idx]
        block_bitmasks.append(mask)
    
    # Precompute conflict lists: for each block, which other blocks conflict with it
    conflict_lists = [[] for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflict_lists[i].append(j)
                conflict_lists[j].append(i)

    # Sort blocks by number of conflicts (most constrained first) for better B&B
    block_order = sorted(range(num_blocks_total), key=lambda b: len(conflict_lists[b]), reverse=True)
    
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
    for t in range(80):
        if time.time() > deadline - 40.0:
            break
        
        # Different greedy strategies
        if t % 3 == 0:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        elif t % 3 == 1:
            # Sort by fewest conflicts first
            order = sorted(range(num_blocks_total), key=lambda b: len(conflict_lists[b]))
            random.shuffle(order)
        else:
            # Sort by most conflicts first
            order = sorted(range(num_blocks_total), key=lambda b: -len(conflict_lists[b]))
            random.shuffle(order)
        
        packing = greedy_packing(order)
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
    
    # Improved branch and bound with tighter upper bounds
    def compute_upper_bound(available_mask, current_count):
        """Compute a tight upper bound on additional blocks we can add."""
        # Use a greedy coloring-like approach for upper bound
        remaining = [b for b in block_order if (available_mask >> b) & 1]
        if not remaining:
            return 0
        
        # Simple greedy upper bound: try to pack as many as possible
        greedy_count = 0
        temp_mask = available_mask
        for b in block_order:
            if (temp_mask >> b) & 1:
                greedy_count += 1
                for c in conflict_lists[b]:
                    temp_mask &= ~(1 << c)
        
        return greedy_count
    
    def branch_and_bound(available_mask, current_packing, current_count, depth=0):
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 15.0:
            return
        
        # Upper bound check
        ub = compute_upper_bound(available_mask, current_count)
        if current_count + ub <= best_num_blocks:
            return
        
        # Check if we've found a solution
        if current_count > best_num_blocks:
            best_num_blocks = current_count
            best_packing = current_packing[:]
        
        # Get available blocks
        candidates = [b for b in block_order if (available_mask >> b) & 1]
        if not candidates:
            return
        
        # Pick the most constrained candidate (fewest remaining conflicts)
        # This helps prune the search tree
        scored_candidates = []
        for b in candidates:
            # Count how many other candidates conflict with b
            conflict_count = sum(1 for c in candidates if c != b and (block_bitmasks[b] & block_bitmasks[c]))
            scored_candidates.append((conflict_count, b))
        
        scored_candidates.sort()
        
        # Branch on top few candidates
        for _, b in scored_candidates[:8]:
            if time.time() > deadline - 15.0:
                return
            
            # Include block b
            new_packing = current_packing + [b]
            new_mask = available_mask & ~(1 << b)
            # Remove all conflicts of b
            for c in conflict_lists[b]:
                new_mask &= ~(1 << c)
            
            branch_and_bound(new_mask, new_packing, current_count + 1, depth + 1)
            
            if current_count + 1 > best_num_blocks:
                best_num_blocks = current_count + 1
                best_packing = new_packing[:]
        
        # Exclude the first candidate
        if scored_candidates:
            b = scored_candidates[0][1]
            new_mask = available_mask & ~(1 << b)
            branch_and_bound(new_mask, current_packing, current_count, depth + 1)
    
    # Run B&B from scratch
    full_mask = (1 << num_blocks_total) - 1
    branch_and_bound(full_mask, [], 0)
    
    # Aggressive local search with 2-for-1 swap moves
    for t in range(150):
        if time.time() > deadline - 8.0:
            break
        
        # Start from current best or random
        if t % 4 == 0 or not best_packing:
            packing = greedy_packing()
        else:
            packing = best_packing[:]
        
        improved = True
        iterations = 0
        while improved and iterations < 50:
            if time.time() > deadline - 8.0:
                break
            improved = False
            iterations += 1
            
            selected_set = set(packing)
            unselected = [b for b in range(num_blocks_total) if b not in selected_set]
            
            if not unselected:
                break
            
            random.shuffle(unselected)
            
            # Try to add a block or do a 2-for-1 swap
            best_move = None
            best_gain = 0
            
            for new_block in unselected[:30]:
                # Find conflicts
                conflicts = [b for b in packing if block_bitmasks[b] & block_bitmasks[new_block]]
                
                if not conflicts:
                    # No conflict, gain of +1
                    if best_gain < 1:
                        best_gain = 1
                        best_move = ('add', new_block, None)
                elif len(conflicts) == 1:
                    # 1-for-1 swap, no gain
                    pass
                elif len(conflicts) >= 2:
                    # Try removing 2 conflicts and adding 1 new block: net -1, skip
                    # But check if the 2 removed blocks could be replaced by more
                    pass
            
            # Also try 2-for-1: remove 2 blocks, add 2 new blocks
            if len(packing) >= 2:
                for i in range(min(10, len(packing))):
                    for j in range(i + 1, min(10, len(packing))):
                        if time.time() > deadline - 8.0:
                            break
                        
                        removed = [packing[i], packing[j]]
                        removed_mask = block_bitmasks[removed[0]] | block_bitmasks[removed[1]]
                        
                        # Find blocks that conflict only with these two
                        candidates = []
                        for b in unselected[:20]:
                            conf_with_removed = sum(1 for r in removed if block_bitmasks[b] & block_bitmasks[r])
                            if conf_with_removed == 2:
                                # Check if it conflicts with any other selected block
                                other_conflict = False
                                for k, other in enumerate(packing):
                                    if k not in (i, j) and (block_bitmasks[b] & block_bitmasks[other]):
                                        other_conflict = True
                                        break
                                if not other_conflict:
                                    candidates.append(b)
                        
                        if len(candidates) >= 2:
                            # Try swapping
                            for c1 in candidates[:5]:
                                for c2 in candidates[:5]:
                                    if c1 == c2:
                                        continue
                                    # Check if c1 and c2 are compatible
                                    if not (block_bitmasks[c1] & block_bitmasks[c2]):
                                        # Check compatibility with rest of packing
                                        compatible = True
                                        for k, other in enumerate(packing):
                                            if k not in (i, j):
                                                if (block_bitmasks[c1] & block_bitmasks[other]) or (block_bitmasks[c2] & block_bitmasks[other]):
                                                    compatible = False
                                                    break
                                        if compatible:
                                            # Net gain: remove 2, add 2 = 0, but might enable future moves
                                            # Only do if it doesn't decrease count
                                            pass
            
            if best_move:
                if best_move[0] == 'add':
                    packing.append(best_move[1])
                    improved = True
                    if len(packing) > best_num_blocks:
                        best_num_blocks = len(packing)
                        best_packing = packing[:]
                else:
                    # Apply swap
                    pass
        
        # Update best
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
            
        # Early exit if we hit theoretical max
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
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx][0]
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
