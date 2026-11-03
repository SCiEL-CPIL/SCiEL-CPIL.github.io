import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Key improvements:
    1. Precompute all blocks once with their index sets
    2. Use bitmasks for fast disjointness checking
    3. Use a better greedy heuristic: prioritize blocks that cover "rare" indices
    4. Multiple random restarts with different random orderings
    5. Try both pure random shuffle and priority-based ordering
    6. Local search: try to swap in a block that allows adding another block
    7. Simulated annealing style: allow temporary worsening for escape
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

    num_total_blocks = len(all_blocks)
    
    # Precompute a bitmask for all triples
    full_mask = (1 << total_triples) - 1

    # Count how many blocks cover each triple (for rarity heuristic)
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for i, j, k in indices:
            triple_block_count[triple_to_idx(i, j, k)] += 1

    best_num_blocks = 0
    best_packing = []

    def greedy_random(block_list, used_mask, num_blocks_current, packing_current):
        """Greedy selection with random ordering."""
        shuffled = list(block_list)
        random.shuffle(shuffled)
        for params, bm, indices in shuffled:
            if (bm & used_mask) == 0:
                packing_current.append(params)
                used_mask |= bm
                num_blocks_current += 1
        return num_blocks_current, packing_current, used_mask

    def greedy_priority(block_list, used_mask, num_blocks_current, packing_current):
        """Greedy selection prioritizing blocks that cover rare indices."""
        shuffled = list(block_list)
        random.shuffle(shuffled)
        
        # Compute priority: sum of rarity of uncovered indices in the block
        for params, bm, indices in shuffled:
            uncovered_indices = [i for i in indices if (triple_to_idx(i[0], i[1], i[2]) & used_mask) == 0]
            # Rarity = 1 / count of blocks covering this index
            priority = sum(1.0 / triple_block_count[triple_to_idx(i[0], i[1], i[2])] for i in uncovered_indices)
            
            if (bm & used_mask) == 0 and priority > 0:
                packing_current.append(params)
                used_mask |= bm
                num_blocks_current += 1
        
        return num_blocks_current, packing_current, used_mask

    def local_search_improve(packing, used_mask, num_blocks):
        """
        Try to improve the packing by swapping out a block to allow adding another.
        Returns (new_num_blocks, new_packing, new_used_mask) if improved, else original.
        """
        if not packing:
            return num_blocks, packing, used_mask
        
        current_best = num_blocks
        current_best_packing = packing[:]
        current_best_mask = used_mask
        
        # Try removing each block and see if we can add more
        for remove_idx in range(len(packing)):
            if time.time() > deadline - 10.0:
                break
            
            # Remove the block
            removed_params = packing[remove_idx]
            removed_bm = all_blocks[[b[0] for b in all_blocks].index(removed_params)][1]
            
            new_mask = used_mask & ~removed_bm
            new_packing = packing[:remove_idx] + packing[remove_idx+1:]
            
            # Try to add any block that fits
            added = False
            for other_params, other_bm, other_indices in all_blocks:
                if other_params in new_packing:
                    continue
                if (other_bm & new_mask) == 0:
                    new_packing.append(other_params)
                    new_mask |= other_bm
                    added = True
                    break
            
            if added and len(new_packing) > current_best:
                current_best = len(new_packing)
                current_best_packing = new_packing
                current_best_mask = new_mask
        
        return current_best, current_best_packing, current_best_mask

    num_trials = 3000
    trial_count = 0
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        trial_count += 1
        
        if t % 3 == 0:
            # Random greedy
            num_blocks, packing, _ = greedy_random(all_blocks, 0, 0, [])
        elif t % 3 == 1:
            # Priority-based greedy
            num_blocks, packing, _ = greedy_priority(all_blocks, 0, 0, [])
        else:
            # Random greedy with local search
            num_blocks, packing, used_mask = greedy_random(all_blocks, 0, 0, [])
            # Apply local search if we have time
            if time.time() < deadline - 15.0 and num_blocks >= best_num_blocks:
                num_blocks, packing, used_mask = local_search_improve(packing, used_mask, num_blocks)
        
        if num_blocks > best_num_blocks:
            best_num_blocks = num_blocks
            best_packing = packing
            
            # Early exit if we hit a good number
            if best_num_blocks >= 12:
                break

    # Also try a more systematic approach: try to find maximum packing
    # by considering blocks in order of increasing overlap with others
    if time.time() < deadline - 10.0:
        # Compute pairwise overlap
        num_blocks_list = len(all_blocks)
        
        # Sort blocks by number of conflicts (fewer conflicts first)
        block_conflicts = []
        for idx, (params, bm, indices) in enumerate(all_blocks):
            conflicts = 0
            for other_idx in range(idx + 1, num_blocks_list):
                if (bm & all_blocks[other_idx][1]) != 0:
                    conflicts += 1
            block_conflicts.append((conflicts, idx))
        
        block_conflicts.sort()
        sorted_block_indices = [idx for _, idx in block_conflicts]
        
        # Greedy on sorted order
        used_mask = 0
        packing = []
        for idx in sorted_block_indices:
            params, bm, indices = all_blocks[idx]
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing

    # Additional: try to find a better packing using a different strategy
    # Focus on the 3 i-dimension: we can only have 1 pair of i's (since n=3)
    # So all blocks must use i1=0, i2=1 or i1=1, i2=2
    # This means blocks with i1=0,i2=1 and i1=1,i2=2 can coexist
    # but two blocks with i1=0,i2=1 cannot coexist (they share i=0 and i=1)
    # Similarly for j and k dimensions
    
    # Let's try a more targeted search
    if time.time() < deadline - 5.0:
        # Group blocks by their i-pair
        blocks_by_i = {
            (0, 1): [],
            (1, 2): []
        }
        for params, bm, indices in all_blocks:
            i1, i2 = params[0], params[1]
            if i1 == 0 and i2 == 1:
                blocks_by_i[(0, 1)].append((params, bm, indices))
            elif i1 == 1 and i2 == 2:
                blocks_by_i[(1, 2)].append((params, bm, indices))
        
        # For each combination of one block from each i-group, check if they're disjoint
        # and try to add more blocks
        for i_pair1_blocks in [blocks_by_i[(0, 1)], [None]]:
            for i_pair2_blocks in [blocks_by_i[(1, 2)], [None]]:
                if time.time() > deadline - 5.0:
                    break
                
                for b1 in i_pair1_blocks:
                    if b1 is None:
                        continue
                    for b2 in i_pair2_blocks:
                        if b2 is None:
                            continue
                        
                        # Check if b1 and b2 are disjoint
                        if (b1[1] & b2[1]) == 0:
                            # Try to build a packing starting with these two
                            used_mask = b1[1] | b2[1]
                            packing = [b1[0], b2[0]]
                            
                            # Add more blocks greedily
                            shuffled = list(all_blocks)
                            random.shuffle(shuffled)
                            for params, bm, indices in shuffled:
                                if params in packing:
                                    continue
                                if (bm & used_mask) == 0:
                                    packing.append(params)
                                    used_mask |= bm
                            
                            if len(packing) > best_num_blocks:
                                best_num_blocks = len(packing)
                                best_packing = packing

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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
