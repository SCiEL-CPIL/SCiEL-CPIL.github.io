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
    
    # Precompute t-subsets for all current blocks
    block_tsubs = [set(itertools.combinations(b, t)) for b in current_blocks]
    
    # Count coverage for each t-subset
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
    
    max_iter = 8000
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
        
        # Try several random new points, pick the one that creates most redundancy
        best_new_point = -1
        best_new_tsubs = None
        best_lost_count = float('inf')
        best_gained_count = 0
        
        for _ in range(15):
            if time.time() > deadline - 5:
                break
                
            new_point = rng.randint(0, v - 1)
            if new_point in block:
                continue
            
            # Candidate new block
            new_block = block.copy()
            new_block[pos] = new_point
            new_block.sort()
            
            old_tsubs = block_tsubs[block_idx]
            new_tsubs = set(itertools.combinations(new_block, t))
            
            # t-subsets that will lose coverage from this block
            lost_tsubs = old_tsubs - new_tsubs
            gained_tsubs = new_tsubs - old_tsubs
            
            # Check if any lost_tsub is covered ONLY by this block
            valid = True
            for ts in lost_tsubs:
                if cover_count[ts] <= 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Score: prefer swaps that create more redundancy (higher cover_count for gained tsubs)
            # and minimize unique coverage loss
            lost_unique = sum(1 for ts in lost_tsubs if cover_count[ts] == 1)
            gained_redundant = sum(1 for ts in gained_tsubs if cover_count[ts] >= 1)
            
            # We want to minimize lost_unique and maximize gained_redundant
            score = lost_unique * 10 - gained_redundant
            if best_new_point == -1 or score < best_lost_count:
                best_new_point = new_point
                best_new_tsubs = new_tsubs
                best_lost_count = score
                best_gained_count = gained_redundant
        
        if best_new_point != -1:
            # Accept swap
            old_tsubs = block_tsubs[block_idx]
            lost_tsubs = old_tsubs - best_new_tsubs
            gained_tsubs = best_new_tsubs - old_tsubs
            
            for ts in lost_tsubs:
                cover_count[ts] -= 1
            for ts in gained_tsubs:
                cover_count[ts] += 1
            
            new_block = block.copy()
            new_block[pos] = best_new_point
            new_block.sort()
            current_blocks[block_idx] = new_block
            block_tsubs[block_idx] = best_new_tsubs
        
        # After a swap, check if we can remove a block
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
                break
        
        if not removed_any:
            # No swap was made or no block removed; continue
            pass

    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 130  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with different seeds
    max_restarts = 40
    for restart in range(max_restarts):
        if time.time() > deadline - 10:  # Leave time for final polish
            break
        
        # Use different random seed for each restart to explore different solutions
        restart_rng = random.Random(42 + restart * 7)
        
        blocks = _greedy_cover(v, k, t, restart_rng, deadline)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        blocks = _local_search(blocks, v, k, t, restart_rng, deadline)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if we hit a very good solution
        if best_count <= 74:
            break
    
    if best_blocks is not None:
        # Final polish: try more aggressive local search on the best solution
        if time.time() < deadline - 15:
            polish_rng = random.Random(999)
            polished = _local_search(best_blocks, v, k, t, polish_rng, deadline)
            if _verify_covering(polished, v, t) and len(polished) < best_count:
                best_blocks = polished
                best_count = len(polished)
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = next(x for x in range(v) if x not in block)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
