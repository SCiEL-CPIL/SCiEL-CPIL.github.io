import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses greedy search with local improvement to find maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks_total = len(all_blocks)
    
    # Precompute index-to-bit mapping for faster conflict checking
    all_indices = set()
    for _, indices in all_blocks:
        all_indices |= indices
    index_to_bit = {idx: bit for bit, idx in enumerate(all_indices)}
    num_bits = len(all_indices)
    
    # Convert blocks to bitmasks
    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << index_to_bit[idx])
        block_bitmasks.append(mask)
    
    # Build conflict graph using bitmasks
    conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    degrees = [len(conflicts[i]) for i in range(num_blocks_total)]
    
    # Precompute non-conflict sets for each block
    non_conflicts = [set(range(num_blocks_total)) - conflicts[i] - {i} for i in range(num_blocks_total)]
    
    def greedy_packing(order):
        """Greedy packing given an order of blocks using bitmask."""
        used_mask = 0
        current = []
        for idx in order:
            if not (block_bitmasks[idx] & used_mask):
                current.append(idx)
                used_mask |= block_bitmasks[idx]
        return current
    
    def get_used_mask(packing):
        """Get the union bitmask of all blocks in a packing."""
        mask = 0
        for idx in packing:
            mask |= block_bitmasks[idx]
        return mask
    
    def local_improve(packing):
        """Try to improve packing by swapping out blocks for better combinations using bitmasks."""
        if not packing:
            return packing
        
        best_packing = packing[:]
        best_mask = get_used_mask(best_packing)
        
        improved = True
        while improved:
            improved = False
            
            # Try removing 1 block and adding 2
            for pos in range(len(best_packing)):
                if time.time() > deadline - 2.0:
                    break
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                used_remaining_mask = get_used_mask(remaining)
                
                # Find blocks that could be added
                candidates = []
                for idx in range(num_blocks_total):
                    if idx in remaining:
                        continue
                    if not (block_bitmasks[idx] & used_remaining_mask):
                        candidates.append(idx)
                
                # Try adding 2 compatible candidates
                best_pair = None
                for i in range(len(candidates)):
                    c1 = candidates[i]
                    used_c1_mask = used_remaining_mask | block_bitmasks[c1]
                    for j in range(i + 1, len(candidates)):
                        c2 = candidates[j]
                        if not (block_bitmasks[c2] & used_c1_mask):
                            best_pair = (c1, c2)
                            break
                    if best_pair:
                        break
                
                if best_pair:
                    new_packing = remaining + [best_pair[0], best_pair[1]]
                    if len(new_packing) > len(best_packing):
                        best_packing = new_packing
                        best_mask = get_used_mask(best_packing)
                        improved = True
                        break
            
            # Try removing 2 blocks and adding 3
            if not improved and len(best_packing) >= 2:
                for pos1 in range(len(best_packing)):
                    if time.time() > deadline - 2.0:
                        break
                    for pos2 in range(pos1 + 1, len(best_packing)):
                        remaining = [x for i, x in enumerate(best_packing) if i != pos1 and i != pos2]
                        used_remaining_mask = get_used_mask(remaining)
                        
                        # Find blocks that could be added
                        candidates = []
                        for idx in range(num_blocks_total):
                            if idx in remaining:
                                continue
                            if not (block_bitmasks[idx] & used_remaining_mask):
                                candidates.append(idx)
                        
                        # Try adding 3 compatible candidates
                        best_triple = None
                        for i in range(len(candidates)):
                            c1 = candidates[i]
                            used_c1_mask = used_remaining_mask | block_bitmasks[c1]
                            for j in range(i + 1, len(candidates)):
                                c2 = candidates[j]
                                if not (block_bitmasks[c2] & used_c1_mask):
                                    used_c2_mask = used_c1_mask | block_bitmasks[c2]
                                    for k in range(j + 1, len(candidates)):
                                        c3 = candidates[k]
                                        if not (block_bitmasks[c3] & used_c2_mask):
                                            best_triple = (c1, c2, c3)
                                            break
                                    if best_triple:
                                        break
                            if best_triple:
                                break
                        
                        if best_triple:
                            new_packing = remaining + list(best_triple)
                            if len(new_packing) > len(best_packing):
                                best_packing = new_packing
                                best_mask = get_used_mask(best_packing)
                                improved = True
                                break
                    if improved:
                        break
        
        return best_packing
    
    def final_polish(packing):
        """Try to add any remaining compatible blocks to the packing."""
        if not packing:
            return packing
        
        result = packing[:]
        result_mask = get_used_mask(result)
        
        # Try to add any block that doesn't conflict
        for idx in range(num_blocks_total):
            if idx not in result and not (block_bitmasks[idx] & result_mask):
                result.append(idx)
                result_mask |= block_bitmasks[idx]
        
        return result
    
    # Initial ordering: sort by degree ascending
    base_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
    
    best_packing = []
    best_size = 0
    
    # Phase 1: Greedy with random restarts and simulated annealing style perturbations
    num_trials = 2000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 10.0:
            break
        
        # Create a perturbed order with varying perturbation strength
        order = base_order[:]
        num_swaps = random.randint(0, 8)
        for _ in range(num_swaps):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        packing = greedy_packing(order)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]
            
            # Local improvement
            if time.time() < deadline - 5.0:
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
            
            # Final polish
            if time.time() < deadline - 3.0:
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
            
            if best_size >= 12:
                break
        
        trial += 1
    
    # Phase 1.5: If we found a good packing, try to improve it further with multiple local improvement passes
    if time.time() < deadline - 5.0 and best_size >= 5:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            improved = local_improve(best_packing)
            if len(improved) > best_size:
                best_size = len(improved)
                best_packing = improved
            else:
                # Try final polish
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
                else:
                    break
    
    # Phase 2: More targeted search if time remains
    if time.time() < deadline - 5.0 and best_size < 12:
        # Try all pairs of blocks and see if they can be extended
        for i in range(num_blocks_total):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, num_blocks_total):
                if block_bitmasks[i] & block_bitmasks[j]:
                    continue
                # Start with {i, j} and greedily extend
                used_mask = block_bitmasks[i] | block_bitmasks[j]
                current = [i, j]
                for k in base_order:
                    if k == i or k == j:
                        continue
                    if not (block_bitmasks[k] & used_mask):
                        current.append(k)
                        used_mask |= block_bitmasks[k]
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_packing = current[:]
                    # Apply local improvement
                    if time.time() < deadline - 5.0:
                        improved = local_improve(best_packing)
                        if len(improved) > best_size:
                            best_size = len(improved)
                            best_packing = improved
                    # Final polish
                    if time.time() < deadline - 3.0:
                        polished = final_polish(best_packing)
                        if len(polished) > best_size:
                            best_size = len(polished)
                            best_packing = polished
                    if best_size >= 12:
                        break
            if best_size >= 12:
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
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
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
