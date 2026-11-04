import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_indices_list = []
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
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices_list.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Pre-compute conflict bitmasks for fast checking
    # Each block conflicts with another if they share any (i,j,k) index
    # Map each (i,j,k) to a unique index
    index_to_id = {}
    all_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_indices.append((i, j, k))
                index_to_id[(i, j, k)] = len(all_indices) - 1
    num_indices = len(all_indices)
    
    # Represent each block as a bitmask of which indices it uses
    # Use Python ints for arbitrary-size bitmasks
    block_masks = []
    for indices in block_indices_list:
        mask = 0
        for idx in indices:
            mask |= (1 << index_to_id[idx])
        block_masks.append(mask)
    
    # Conflict: blocks i and j conflict if (block_masks[i] & block_masks[j]) != 0
    # Pre-compute conflict sets
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if (block_masks[i] & block_masks[j]) != 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Also pre-compute for each block, the set of non-conflicting blocks
    non_conflicts = [set(range(num_blocks)) - {i} - conflicts[i] for i in range(num_blocks)]
    
    def local_search(packing):
        """
        Improve packing by trying to add blocks, or swap blocks to allow more blocks.
        """
        packing = list(packing)
        used = set(packing)
        improved = True
        iteration = 0
        while improved and iteration < 50:
            improved = False
            iteration += 1
            
            # 1. Try to add a block that fits
            for b in range(num_blocks):
                if b in used:
                    continue
                if not (conflicts[b] & used):
                    packing.append(b)
                    used.add(b)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1->2 swap (remove 1, add 2)
                found_swap = False
                b1_order = list(packing)
                random.shuffle(b1_order)
                
                for b1 in b1_order:
                    used_temp = used - {b1}
                    # Find blocks that don't conflict with used_temp
                    candidates = [b for b in range(num_blocks) if b not in used_temp and not (conflicts[b] & used_temp)]
                    
                    pair_found = None
                    for i in range(len(candidates)):
                        c1 = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            c2 = candidates[j]
                            if c2 not in conflicts[c1]:
                                pair_found = (c1, c2)
                                break
                        if pair_found:
                            break
                    
                    if pair_found:
                        new_packing = [b for b in packing if b != b1]
                        new_packing.extend(pair_found)
                        packing = new_packing
                        used = set(packing)
                        improved = True
                        found_swap = True
                        break
                
                if not found_swap:
                    # 3. Try 1->3 swap (remove 1, add 3)
                    # Net gain +2. Very powerful but rare.
                    b1_order = list(packing)
                    random.shuffle(b1_order)
                    
                    for b1 in b1_order:
                        used_temp = used - {b1}
                        candidates = [b for b in range(num_blocks) if b not in used_temp and not (conflicts[b] & used_temp)]
                        
                        triple_found = None
                        # Limit search to small candidate sets to avoid O(M^3) explosion
                        if len(candidates) <= 40:
                            for i in range(len(candidates)):
                                c1 = candidates[i]
                                for j in range(i + 1, len(candidates)):
                                    c2 = candidates[j]
                                    if c2 in conflicts[c1]: continue
                                    for k in range(j + 1, len(candidates)):
                                        c3 = candidates[k]
                                        if c3 in conflicts[c1] or c3 in conflicts[c2]: continue
                                        triple_found = (c1, c2, c3)
                                        break
                                    if triple_found: break
                                if triple_found: break
                        
                        if triple_found:
                            new_packing = [b for b in packing if b != b1]
                            new_packing.extend(triple_found)
                            packing = new_packing
                            used = set(packing)
                            improved = True
                            break
                    
                    if not improved:
                        # 4. Try 2->3 swap (remove 2, add 3)
                        # Net gain +1
                        b12_order = []
                        packing_list = list(packing)
                        random.shuffle(packing_list)
                        for i in range(len(packing_list)):
                            for j in range(i+1, len(packing_list)):
                                b12_order.append((packing_list[i], packing_list[j]))
                        random.shuffle(b12_order)
                        
                        for b1, b2 in b12_order:
                            used_temp = used - {b1, b2}
                            candidates = [b for b in range(num_blocks) if b not in used_temp and not (conflicts[b] & used_temp)]
                            
                            triple_found = None
                            if len(candidates) <= 50:
                                for i in range(len(candidates)):
                                    c1 = candidates[i]
                                    for j in range(i + 1, len(candidates)):
                                        c2 = candidates[j]
                                        if c2 in conflicts[c1]: continue
                                        for k in range(j + 1, len(candidates)):
                                            c3 = candidates[k]
                                            if c3 in conflicts[c1] or c3 in conflicts[c2]: continue
                                            triple_found = (c1, c2, c3)
                                            break
                                        if triple_found: break
                                    if triple_found: break
                            
                            if triple_found:
                                new_packing = [b for b in packing if b != b1 and b != b2]
                                new_packing.extend(triple_found)
                                packing = new_packing
                                used = set(packing)
                                improved = True
                                break

        return packing

    best_packing = []
    best_size = 0
    
    # Multiple random restarts with different strategies
    num_restarts = 3000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(r)
        current_packing = []
        used = set()
        
        # Try different greedy strategies
        strategy = r % 3
        available = list(range(num_blocks))
        
        if strategy == 0:
            random.shuffle(available)
        elif strategy == 1:
            # Sort by number of conflicts (prefer less constrained)
            available.sort(key=lambda b: len(conflicts[b]))
        else:
            # Sort by degree in conflict graph, reverse
            available.sort(key=lambda b: -len(conflicts[b]))
            random.shuffle(available)
        
        for b in available:
            if b in used:
                continue
            if not (conflicts[b] & used):
                current_packing.append(b)
                used.add(b)
        
        # Local search
        current_packing = local_search(current_packing)
        
        if len(current_packing) > best_size:
            best_size = len(current_packing)
            best_packing = list(current_packing)
            
            # Early exit if we hit theoretical max
            if best_size >= 13:
                break

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
