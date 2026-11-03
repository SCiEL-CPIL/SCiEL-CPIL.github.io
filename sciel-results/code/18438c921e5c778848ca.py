"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute(v, k, t):
    """Precompute subset mappings and adjacency structures for fast gain calculation."""
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map subset (as frozenset) to index
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    
    # For t=3, precompute for each pair (a,b) the list of subset indices containing {a,b}
    # This allows O(1) lookup of which triples are formed by adding x to a pair {a,b}
    pair_to_subsets = {}
    for i, s in enumerate(all_subsets):
        # s is a tuple of 3 elements
        a, b, c = s
        # Add to pairs (a,b), (a,c), (b,c)
        pair_to_subsets.setdefault((a, b), []).append(i)
        pair_to_subsets.setdefault((a, c), []).append(i)
        pair_to_subsets.setdefault((b, c), []).append(i)
        
    return all_subsets, subset_to_idx, pair_to_subsets, n_subsets


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets):
    """Fast greedy construction using precomputed pair-to-subset mappings."""
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        block_set_local = set(block)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # For each pair in current block, check triples formed with x
                for a, b in itertools.combinations(block, 2):
                    # Look up triples containing a, b
                    # The triple must contain x, so it is {a, b, x}
                    # We can just check if {a, b, x} is in subset_to_idx and in uncovered
                    # But using pair_to_subsets is faster if we precomputed it for pairs?
                    # Actually, for t=3, the triple is uniquely determined by {a,b,x}.
                    # So we just need to check if frozenset({a,b,x}) is in subset_to_idx.
                    # Since v is small, direct check is fast enough, but let's use the map.
                    fs = frozenset((a, b, x))
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
            block_set_local.add(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            # Update uncovered
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks


def _block_covered_ids(block, subset_to_idx, t):
    """Get list of subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, t):
        covered.append(subset_to_idx[frozenset(c)])
    return covered


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Precompute
    all_subsets, subset_to_idx, pair_to_subsets, n_subsets = _precompute(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets)
    
    # Precompute covered subset IDs for each block
    block_covered = [_block_covered_ids(b, subset_to_idx, t) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            
            # Identify critical subsets (count == 1) in old_block
            critical = [idx for idx in old_covered if counts[idx] == 1]
            
            # Collect points needed for critical subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_subsets[idx])
            
            # If we can't fit all critical points in one block, skip
            if len(points_needed) > k:
                continue
            
            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedy fill to maximize coverage of low-count subsets
            # We want to cover as many subsets with count 0 or 1 as possible
            # But we must keep the block size k
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                # Calculate gain: how many subsets with count 0 or 1 does adding x cover?
                # Subsets formed by x and pairs in new_block_set
                for a, b in itertools.combinations(new_block_set, 2):
                    fs = frozenset((a, b, x))
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if counts[idx] == 0:
                            gain += 10
                        elif counts[idx] == 1:
                            gain += 1
                
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                
                if new_block == old_block:
                    continue
                
                new_covered = _block_covered_ids(new_block, subset_to_idx, t)
                
                # Check if swap is valid: no subset loses its last cover
                valid = True
                old_covered_set = set(old_covered)
                new_covered_set = set(new_covered)
                
                for idx in old_covered:
                    if idx not in new_covered and counts[idx] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    for idx in old_covered:
                        counts[idx] -= 1
                    for idx in new_covered:
                        counts[idx] += 1
                    
                    blocks[i] = new_block
                    block_covered[i] = new_covered
                    
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
