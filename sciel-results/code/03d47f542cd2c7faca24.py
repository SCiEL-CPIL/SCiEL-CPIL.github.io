import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
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

    num_blocks = len(all_blocks)
    
    # Build conflict graph
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    degrees = [len(conflicts[i]) for i in range(num_blocks)]
    
    # Helper to check if a set of block indices is independent
    def is_independent(block_indices):
        for i in range(len(block_indices)):
            for j in range(i+1, len(block_indices)):
                if block_indices[j] in conflicts[block_indices[i]]:
                    return False
        return True

    # Greedy packing
    def greedy_packing(order):
        used = set()
        current = []
        for idx in order:
            if not (all_blocks[idx][1] & used):
                current.append(idx)
                used |= all_blocks[idx][1]
        return current

    # Local improvement: try to swap one block for two
    def local_improve(packing):
        if not packing:
            return packing
        
        best_packing = packing[:]
        improved = True
        while improved:
            improved = False
            for pos in range(len(best_packing)):
                # Try removing block at pos
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                
                # Try adding any single block that fits
                # and then see if we can add another
                # This is O(N^2) per iteration which might be slow, 
                # but N is small (max independent set size is small).
                
                # Actually, a simpler local search:
                # Try removing a block and replacing it with two non-conflicting blocks.
                
                # Let's try a simpler heuristic: 
                # For each block in the packing, try to remove it.
                # Then run greedy on the remaining available blocks (excluding those conflicting with the rest of the packing).
                # If the new size is larger, keep it.
                
                # However, implementing a full local search with backtracking is complex.
                # Let's stick to random restarts with a good initial ordering, 
                # and maybe a simple "swap" improvement.
                
                # Let's try: for each block in packing, try to remove it.
                # Then, from the blocks NOT in packing and NOT conflicting with (packing \ {removed}),
                # greedily pick blocks that fit.
                # If the resulting size is > len(packing), update.
                
                # This is O(N^2) per step. With N~100, this is 10000 ops per step.
                # Packing size ~10. Steps ~10. Total ~100k ops per local improve call.
                # We have ~100 restarts. 10M ops. Should be fast enough.
                
                available = []
                for b in range(num_blocks):
                    if b in best_packing:
                        continue
                    if b == removed:
                        continue
                    # Check if b conflicts with any block in remaining
                    conflict = False
                    for r in remaining:
                        if b in conflicts[r]:
                            conflict = True
                            break
                    if not conflict:
                        available.append(b)
                
                # Greedily pick from available
                # Sort by degree or random? Random is fine for diversity.
                random.shuffle(available)
                
                new_packing = remaining[:]
                used = set()
                for b in remaining:
                    used |= all_blocks[b][1]
                
                for b in available:
                    if not (all_blocks[b][1] & used):
                        new_packing.append(b)
                        used |= all_blocks[b][1]
                
                if len(new_packing) > len(best_packing):
                    best_packing = new_packing
                    improved = True
                    break # Restart the while loop with new packing
        
        return best_packing

    best_packing = []
    best_size = 0

    # Multiple restarts
    num_restarts = 500
    for _ in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        # Heuristic ordering: sort by degree (ascending) to pick less constrained blocks first
        # Then add some randomness
        order = list(range(num_blocks))
        # Sort by degree, but add random noise to break ties
        random.shuffle(order)
        order.sort(key=lambda x: (degrees[x], random.random()))
        
        packing = greedy_packing(order)
        
        # Local improvement
        packing = local_improve(packing)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing
            
            if best_size == 13: # Theoretical max for 3x5x7?
                # Max disjoint 2x2x2 blocks.
                # Volume is 3*5*7 = 105. Each block covers 8.
                # 105/8 = 13.125. So 13 is possible?
                # But dimensions are small.
                # i: 3 -> 1 pair (0,1) or (1,2) or (0,2). Only 1 disjoint pair of i's possible? No, (0,1) and (1,2) share 1.
                # So for i-dim 3, we can have at most 1 block if we require disjoint i's?
                # Wait, "disjoint" means the sets of indices (i,j,k) are disjoint.
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                # Block 2: i in {1,2}, j in {2,3}, k in {2,3}
                # Intersection of indices:
                # i=1 is in both. But j=0,1 vs j=2,3 are disjoint. So no (i,j,k) overlap.
                # So yes, we can have multiple blocks.
                pass

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
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for idx in best_packing:
        params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = params
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
