import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Pre-compute: for each possible block (k-subset), which triples it covers
    # Store as list of arrays of triple indices
    all_blocks_list = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks_list)
    n_per_block = math_comb(k, t)
    
    def math_comb(n, r):
        import math
        return math.comb(n, r)
    
    # Pre-compute block_masks: for each block, a frozenset of triple indices it covers
    # This is memory heavy but enables fast operations
    block_triples = []
    for blk in all_blocks_list:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(blk, t)]
        block_triples.append(frozenset(tris))
    
    # Also pre-compute as numpy arrays for faster bitwise ops
    # Use a 2D boolean matrix: n_blocks x n_triples
    # n_blocks=346104, n_triples=2024 -> 346104*2024 bool = ~700MB. Too much.
    # Instead, keep as list of sets.
    
    # Block index by tuple
    block_to_idx = {blk: i for i, blk in enumerate(all_blocks_list)}
    
    def greedy_global():
        """Global greedy: at each step, pick the block covering most uncovered triples."""
        uncovered = set(range(n_triples))
        chosen_blocks = []
        chosen_set = set()
        
        while uncovered and time.time() < deadline - 20:
            best_gain = 0
            best_block = None
            best_idx = None
            
            for idx, blk in enumerate(all_blocks_list):
                if idx in chosen_set:
                    continue
                # Count how many uncovered triples this block covers
                gain = len(block_triples[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = blk
                    best_idx = idx
            
            if best_block is None or best_gain == 0:
                break
            
            chosen_blocks.append(best_block)
            chosen_set.add(best_idx)
            uncovered -= block_triples[best_idx]
        
        return [list(b) for b in chosen_blocks]
    
    def remove_redundant(blocks):
        """Remove any redundant blocks."""
        if not blocks:
            return blocks
        n = len(blocks)
        # Count coverage
        counts = [0] * n_triples
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                redundant = True
                for tri in itertools.combinations(b, t):
                    if counts[triple_to_idx[tri]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for tri in itertools.combinations(b, t):
                        counts[triple_to_idx[tri]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    def local_search_2for1(blocks):
        """Try to replace 2 blocks with 1 block that covers all their unique triples."""
        blocks = list(blocks)
        n = len(blocks)
        
        # Pre-compute coverage counts
        def compute_counts(blks):
            counts = [0] * n_triples
            for b in blks:
                for tri in itertools.combinations(b, t):
                    counts[triple_to_idx[tri]] += 1
            return counts
        
        counts = compute_counts(blocks)
        
        while time.time() < deadline - 10:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(blocks)):
                    if time.time() > deadline - 10:
                        break
                    
                    # Simulate removing blocks i and j
                    # Find triples that become uncovered
                    lost = []
                    for tri in itertools.combinations(blocks[i], t):
                        ti = triple_to_idx[tri]
                        if counts[ti] == 1:
                            lost.append(ti)
                    for tri in itertools.combinations(blocks[j], t):
                        ti = triple_to_idx[tri]
                        if counts[ti] == 1:
                            lost.append(ti)
                        elif counts[ti] == 2:
                            # Was covered by both i and j, now uncovered
                            lost.append(ti)
                    
                    # Remove duplicates
                    lost = list(set(lost))
                    
                    if not lost:
                        # Both blocks are redundant
                        blocks.pop(j)
                        blocks.pop(i)
                        counts = compute_counts(blocks)
                        improved = True
                        break
                    
                    # Try to find a single block that covers all lost triples
                    # The block must contain the union of all points in lost triples
                    U_points = set()
                    for ti in lost:
                        U_points.update(all_triples[ti])
                    
                    if len(U_points) > k:
                        continue
                    
                    # Find a block that contains U_points
                    remaining = [x for x in range(v) if x not in U_points]
                    need = k - len(U_points)
                    
                    found = False
                    for combo in itertools.combinations(remaining, need):
                        candidate = tuple(sorted(list(U_points) + list(combo)))
                        if candidate in block_to_idx:
                            ci = block_to_idx[candidate]
                            # Check it's not already in blocks
                            if not any(b == list(candidate) for b in blocks):
                                # Verify it covers all lost triples
                                c_tris = block_triples[ci]
                                if all(ti in c_tris for ti in lost):
                                    # Replace blocks i and j with candidate
                                    blocks.pop(j)
                                    blocks.pop(i)
                                    blocks.append(list(candidate))
                                    counts = compute_counts(blocks)
                                    improved = True
                                    found = True
                                    break
                    if found:
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try 1-for-1 swaps to escape local minima
                for i in range(len(blocks)):
                    if time.time() > deadline - 10:
                        break
                    counts = compute_counts(blocks)
                    lost = []
                    for tri in itertools.combinations(blocks[i], t):
                        ti = triple_to_idx[tri]
                        if counts[ti] == 1:
                            lost.append(ti)
                    
                    if not lost:
                        blocks.pop(i)
                        counts = compute_counts(blocks)
                        improved = True
                        break
                    
                    if len(lost) > 5:
                        continue
                    
                    U_points = set()
                    for ti in lost:
                        U_points.update(all_triples[ti])
                    
                    if len(U_points) > k:
                        continue
                    
                    remaining = [x for x in range(v) if x not in U_points]
                    need = k - len(U_points)
                    
                    for combo in itertools.combinations(remaining, need):
                        candidate = tuple(sorted(list(U_points) + list(combo)))
                        if candidate in block_to_idx:
                            ci = block_to_idx[candidate]
                            if not any(b == list(candidate) for b in blocks):
                                c_tris = block_triples[ci]
                                if all(ti in c_tris for ti in lost):
                                    blocks[i] = list(candidate)
                                    counts = compute_counts(blocks)
                                    improved = True
                                    break
                    if improved:
                        break
        
        return remove_redundant(blocks)
    
    # 1. Global greedy construction
    blocks = greedy_global()
    
    # 2. Remove redundant blocks
    blocks = remove_redundant(blocks)
    
    # 3. Local search: 2-for-1 elimination and 1-for-1 swaps
    if time.time() < deadline - 10:
        blocks = local_search_2for1(blocks)
    
    # 4. Final cleanup
    blocks = remove_redundant(blocks)
    
    return blocks
