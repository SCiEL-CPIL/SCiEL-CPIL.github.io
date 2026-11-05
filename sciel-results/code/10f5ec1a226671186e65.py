import itertools
import time

def construct(v, k, t):
    all_points = range(v)
    
    def bits_to_mask(bits):
        m = 0
        for b in bits:
            m |= (1 << b)
        return m

    def mask_to_points(mask):
        return [i for i in range(v) if (mask >> i) & 1]

    # Pre-compute all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        all_t_masks.append(bits_to_mask(combo))
        
    # Initial state
    uncovered = set(all_t_masks)
    blocks = []
    coverage_count = {m: 0 for m in all_t_masks}

    def add_block(block_pts):
        nonlocal uncovered, blocks, coverage_count
        blocks.append(sorted(block_pts))
        for sub in itertools.combinations(block_pts, t):
            m = bits_to_mask(sub)
            coverage_count[m] += 1
            if coverage_count[m] == 1:
                # It was uncovered, now covered
                if m in uncovered:
                    uncovered.remove(m)

    def remove_block(block_pts):
        nonlocal uncovered, blocks, coverage_count
        # Find index
        idx = blocks.index(sorted(block_pts))
        blocks.pop(idx)
        for sub in itertools.combinations(block_pts, t):
            m = bits_to_mask(sub)
            coverage_count[m] -= 1
            if coverage_count[m] == 0:
                uncovered.add(m)

    def prune_blocks():
        nonlocal uncovered, blocks, coverage_count
        changed = True
        while changed:
            changed = False
            # Iterate backwards to safely remove
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if any t-subset in b has count 1
                essential = False
                for sub in itertools.combinations(b, t):
                    m = bits_to_mask(sub)
                    if coverage_count[m] == 1:
                        essential = True
                        break
                if not essential:
                    remove_block(b)
                    changed = True

    # Phase 1: Deterministic Greedy Construction with Symmetry Breaking
    # We fix the first block to be {0, 1, ..., k-1} to break symmetry
    if k <= v:
        first_block = list(range(k))
        add_block(first_block)
        
        while uncovered:
            # Pick lexicographically smallest uncovered t-subset
            # This requires converting masks to tuples for comparison
            # To speed up, we can just pick min(mask) as a heuristic, 
            # but lex order is better for determinism and symmetry.
            # Given N=11628, min() with key is O(N).
            
            # Optimization: If we just pick min(uncovered), it's faster but less "symmetric"
            # Let's stick to min(uncovered) for speed, as the greedy gain calculation
            # will do the heavy lifting. The choice of start subset matters less
            # if the block construction is strong.
            
            start_mask = min(uncovered)
            start_pts = mask_to_points(start_mask)
            
            block_pts = start_pts[:]
            
            while len(block_pts) < k:
                best_x = -1
                best_gain = -1
                
                # Candidates: points not in block_pts
                # To make it deterministic, iterate 0..v-1
                t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                
                for x in range(v):
                    if x in block_pts:
                        continue
                    
                    gain = 0
                    for sub in t_minus_1_subs:
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        m |= (1 << x)
                        if m in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    # Tie-breaking: choose smallest x (implicit in loop order)
                    
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    # But if all gains are 0, pick smallest available
                    for x in range(v):
                        if x not in block_pts:
                            best_x = x
                            break
                            
                block_pts.append(best_x)
            
            add_block(block_pts)
            
    # Phase 2: Pruning
    prune_blocks()
    
    # Phase 3: Local Search / Improvement
    # Try to remove blocks by finding replacements?
    # For now, just return the pruned greedy result.
    # To improve, we can run this multiple times with different "first blocks"?
    # No, symmetry breaking fixes it.
    # We can try a second pass with a different heuristic for tie-breaking?
    
    # Let's add a simple local search: 
    # Try to remove a block, if it breaks coverage, try to "repair" by adding a better block?
    # This is complex.
    
    # Instead, let's just do a few random restarts of the *pruning*? No, pruning is deterministic.
    
    # Let's try a "Swap" local search.
    # For each block, try swapping one point with another to see if it allows pruning of another block?
    # Too slow.
    
    return blocks
