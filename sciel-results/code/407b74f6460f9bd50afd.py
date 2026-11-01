"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """Greedy covering with randomized block selection using numpy for speed."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a random uncovered t-subset to start a new block
        start_tsub = rng.choice(list(uncovered))
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            
            # Precompute combinations of block of size t-1
            block_combos = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_combos:
                    # Create the t-subset with x added
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # No point improves; pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    # Remove blocks where all their t-subsets are covered by other blocks
    kept_blocks = []
    for i, block in enumerate(blocks):
        all_redundant = True
        for ts in block_tsubs[i]:
            if cover_count[ts] == 1:
                all_redundant = False
                break
        if not all_redundant:
            kept_blocks.append(block)
    
    # Update cover counts for remaining blocks
    if len(kept_blocks) < len(blocks):
        cover_count2 = defaultdict(int)
        for block in kept_blocks:
            for ts in itertools.combinations(block, t):
                cover_count2[ts] += 1
        
        # Second pass: remove blocks that are now redundant
        final_blocks = []
        for i, block in enumerate(kept_blocks):
            all_redundant = True
            for ts in itertools.combinations(block, t):
                if cover_count2[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                final_blocks.append(block)
        
        return final_blocks
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def _local_search(blocks, v, k, t, rng, deadline):
    """Local search: try swapping points in blocks to reduce total block count or improve coverage."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    current_count = len(current_blocks)
    best_blocks = [list(b) for b in current_blocks]
    best_count = current_count
    
    # Precompute mapping from t-subset to index
    # C(24,3) = 2024, which is small enough for direct indexing if we map efficiently
    # Or we can just use a set of tuples for simplicity if we optimize the check.
    # Given the speed requirements, let's use a set of tuples for "uncovered" check logic?
    # No, we need to check if a t-subset is covered by *other* blocks.
    
    # Fast approach:
    # 1. Maintain a set of "uncovered" t-subsets? No, we need to ensure swaps don't uncover anything.
    # 2. Better: Maintain a count of coverage for each t-subset?
    
    # Let's stick to the set-based approach but optimize the "is covered by other" check.
    # For a given block B, and a candidate swap (remove x, add y), we form new block B'.
    # We need to check if every t-subset of B' is covered by some block in CurrentBlocks \ {B}.
    
    # Precompute t-subsets for all current blocks
    block_tsubs = [set(itertools.combinations(b, t)) for b in current_blocks]
    
    # Check initial validity (should be valid)
    # all_tsubs = set(itertools.combinations(range(v), t))
    # covered = set().union(*block_tsubs)
    # if not all_tsubs.issubset(covered): return current_blocks
    
    # Helper to check if a t-subset is covered by blocks excluding index `exclude_idx`
    # This is expensive if done naively.
    # Optimization: Count coverage for each t-subset.
    cover_count = defaultdict(int)
    for ts_set in block_tsubs:
        for ts in ts_set:
            cover_count[ts] += 1
    
    # Try to remove blocks one at a time if they become redundant
    # We can do this before swaps to start with a minimal set
    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
        # Check if block i is redundant (all its t-subsets covered by others)
        is_redundant = True
        for ts in block_tsubs[i]:
            if cover_count[ts] <= 1:
                is_redundant = False
                break
        if is_redundant:
            # Remove block i
            for ts in block_tsubs[i]:
                cover_count[ts] -= 1
            current_blocks.pop(i)
            block_tsubs.pop(i)
            current_count -= 1
            best_blocks = [list(b) for b in current_blocks]
            best_count = current_count
    
    # Local Search Loop
    # We want to swap points to increase redundancy (cover_count > 1 for more t-subsets)
    # which allows removal of blocks.
    
    # Time budget for local search
    # We have deadline. Let's use most of it.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline - 5: # Safety margin
            break
        if not current_blocks:
            break
        
        block_idx = rng.randint(0, len(current_blocks) - 1)
        block = current_blocks[block_idx]
        
        # Try swapping a point in block with a point not in block
        # Pick a random position
        pos = rng.randint(0, len(block) - 1)
        orig_point = block[pos]
        
        # Try a few random new points
        found_swap = False
        for _ in range(10):
            if time.time() > deadline - 5:
                break
                
            new_point = rng.randint(0, v - 1)
            if new_point in block:
                continue
            
            # Candidate new block
            new_block = block.copy()
            new_block[pos] = new_point
            new_block.sort()
            
            # Check validity:
            # 1. All t-subsets of new_block must be covered by OTHER blocks.
            # 2. We don't actually need to check if t-subsets of OLD block are still covered,
            #    because we are replacing the block. The new block covers its own t-subsets.
            #    Wait, if we replace block B with B', we lose coverage of t-subsets in B \ B'
            #    that are NOT in B'.
            #    So we must check:
            #    For every ts in (tsubs(B) U tsubs(B')), is it covered by (AllBlocks \ {B}) U {B'}?
            #    Actually, simpler:
            #    The set of covered t-subsets changes from C_old to C_new.
            #    C_new = (C_old \ tsubs(B)) U tsubs(B')
            #    We require C_new to be a superset of AllTSubsets.
            #    This is equivalent to:
            #    1. All t-subsets in tsubs(B) \ tsubs(B') must be covered by OtherBlocks.
            #    2. All t-subsets in tsubs(B') must be covered by OtherBlocks U {B'} (which is true by definition for B').
            #    So the condition is: Every t-subset in tsubs(B) that is NOT in tsubs(B') must be covered by OtherBlocks.
            
            old_tsubs = block_tsubs[block_idx]
            new_tsubs = set(itertools.combinations(new_block, t))
            
            # t-subsets that will lose coverage from this block
            lost_tsubs = old_tsubs - new_tsubs
            
            # Check if any lost_tsub is covered ONLY by this block
            valid = True
            for ts in lost_tsubs:
                # cover_count[ts] includes this block.
                # So if cover_count[ts] == 1, it is not covered by others.
                if cover_count[ts] <= 1:
                    valid = False
                    break
            
            if valid:
                # Accept swap
                # Update cover_count
                for ts in lost_tsubs:
                    cover_count[ts] -= 1
                for ts in (new_tsubs - old_tsubs):
                    cover_count[ts] += 1
                
                current_blocks[block_idx] = new_block
                block_tsubs[block_idx] = new_tsubs
                found_swap = True
                break
        
        # After a swap, check if we can remove a block
        # We can check all blocks or just random ones. Checking all is O(N * C(k,t))
        # N is ~76, C(7,3) is 35. 76*35 ~ 2660 ops. Very fast.
        removed_any = False
        for i in range(len(current_blocks) - 1, -1, -1):
            is_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for ts in block_tsubs[i]:
                    cover_count[ts] -= 1
                current_blocks.pop(i)
                block_tsubs.pop(i)
                current_count -= 1
                best_blocks = [list(b) for b in current_blocks]
                best_count = current_count
                removed_any = True
                break # Restart removal loop if we removed one, or just continue? 
                # Since indices shift, break and let next iteration handle it, or just continue carefully.
                # Actually, if we pop i, indices < i are safe, indices > i are shifted.
                # Since we iterate backwards, popping i is safe for the current iteration's state of "other" blocks.
                # But we should probably just break and let the next outer loop iteration handle further removals.
                # Or we can just continue the loop, knowing that `current_blocks` changed.
                # Let's just break to be safe and fast.
                break
        
        if not found_swap and not removed_any:
            # Maybe try a different block?
            # We can just continue the outer loop.
            pass

    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 130  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 50  # Increase restarts
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        blocks = _greedy_cover(v, k, t, rng, deadline)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        blocks = _local_search(blocks, v, k, t, rng, deadline)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy (original approach)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
