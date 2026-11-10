"""Covering design C(v,k,t) via bitset-accelerated greedy with local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}
    
    # Precompute for each point x and each (t-1)-tuple, the list of t-subset indices
    # that would be covered by adding x to that (t-1)-tuple
    # gain_table[x][tuple_of_t_minus_1] = list of t-subset indices
    # But this is too large. Instead, we'll compute gain on the fly with bitsets.
    
    # Represent uncovered t-subsets as a Python int bitset
    # bit i is 1 if t-subset i is uncovered
    uncovered = (1 << n_t) - 1
    n_uncovered = n_t
    blocks = []
    
    # For speed, precompute for each point x and each subset S of size t-1,
    # the bitmask of t-subsets that contain x and all elements of S
    # This is: for each (t-1)-subset S not containing x, the t-subsets are S ∪ {x}
    # So we need a mapping from (x, S) to the set of t-subset indices
    # Actually, adding x to a partial block B covers all t-subsets {x} ∪ R where R ⊆ B, |R| = t-1
    # So we need: for each x and each (t-1)-subset R of other points, the bitmask of t-subsets = R ∪ {x}
    
    # Precompute: for each x, for each (t-1)-subset R of [v]\{x}, store the bitmask
    # This might be large but v=24, t=3, so (t-1)=2, and C(23,2)=253 per point, total 24*253=6072 entries
    # Each entry is a bitmask of n_t bits. n_t = C(24,3) = 2024. So each bitmask is 2024 bits = 253 bytes.
    # Total: 6072 * 253 bytes ≈ 1.5 MB. That's fine.
    
    gain_masks = {}
    for x in range(v):
        others = [y for y in range(v) if y != x]
        for R in itertools.combinations(others, t - 1):
            # The t-subsets covered are exactly {x} ∪ R
            # But wait: adding x to block B covers all t-subsets that contain x and (t-1) elements from B
            # So for a given partial block B, the gain is the union over all (t-1)-subsets R of B\{x} of the mask for (x, R)
            # Actually no: the t-subsets covered by adding x to B are those of the form {x} ∪ R where R ⊆ B, |R| = t-1
            # So we need the mask for the specific (t-1)-subset R that is part of B
            ts = tuple(sorted(R + (x,)))
            idx = t_index[ts]
            gain_masks[(x, R)] = (1 << idx)
    
    # For a partial block B (list of points), the gain of adding x is:
    # OR of gain_masks[(x, R)] for all R in C(B, t-1)
    # We can compute this as: for each (t-1)-subset R of B, if x not in R, add gain_masks[(x, tuple(sorted(R)))]
    
    def compute_gain(x, block):
        """Compute bitmask of newly covered t-subsets if x is added to block."""
        mask = 0
        for R in itertools.combinations(block, t - 1):
            R_sorted = tuple(sorted(R))
            if x not in R:
                mask |= gain_masks[(x, R_sorted)]
        return mask
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset to start a new block
        # Find the lowest set bit in uncovered
        lowest = (uncovered & -uncovered).bit_length() - 1
        start_ts = all_t[lowest]
        block = list(start_ts)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain_count = -1
            best_gain_mask = 0
            
            for x in range(v):
                if x in block:
                    continue
                gm = compute_gain(x, block)
                # Count how many uncovered subsets this would cover
                new_cover = gm & uncovered
                cnt = bin(new_cover).count('1')
                if cnt > best_gain_count:
                    best_gain_count = cnt
                    best_gain_mask = new_cover
                    best_x = x
                    if cnt == n_uncovered:  # can't do better
                        break
            
            if best_x == -1:
                break
            
            block.append(best_x)
            # Update uncovered: remove all t-subsets covered by the new full block
            # Actually, we should update after the block is complete, or incrementally
            # Let's update incrementally: when we add x, we cover the gain
            # But the gain depends on the current block, so let's just recompute at the end
        
        if len(block) < k:
            # If we couldn't complete the block, add random points
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        
        # Compute all t-subsets of this block and remove them from uncovered
        block_mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            block_mask |= (1 << idx)
        
        uncovered &= ~block_mask
        n_uncovered = bin(uncovered).count('1')
        blocks.append(block)
    
    # Remove duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)
    
    # Local improvement: try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 10:
        n_blocks = len(unique_blocks)
        # For each t-subset, track which blocks cover it
        ts_to_blocks = defaultdict(list)
        for bi, b in enumerate(unique_blocks):
            for ts in itertools.combinations(b, t):
                ts_to_blocks[ts].append(bi)
        
        # Try to remove blocks one by one
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for bi in range(len(unique_blocks)):
                if time.time() > deadline - 5:
                    break
                # Check if block bi is redundant
                redundant = True
                for ts in itertools.combinations(unique_blocks[bi], t):
                    covering = [bj for bj in ts_to_blocks[ts] if bj != bi]
                    if not covering:
                        redundant = False
                        break
                if redundant:
                    # Remove block bi
                    for ts in itertools.combinations(unique_blocks[bi], t):
                        ts_to_blocks[ts].remove(bi)
                    unique_blocks.pop(bi)
                    changed = True
                    break  # restart scan since indices changed
        
        # Re-index ts_to_blocks after removals
        # (not strictly needed since we're done)
    
    return unique_blocks
# EVOLVE-BLOCK-END
