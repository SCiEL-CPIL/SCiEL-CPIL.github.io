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
    """Local search: swap points to create redundancy, then remove redundant blocks."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    current_count = len(current_blocks)
    best_blocks = [list(b) for b in current_blocks]
    best_count = current_count
    
    # Compute coverage counts for current solution
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Try to remove blocks one at a time if they become redundant
    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
        is_redundant = True
        for ts in itertools.combinations(current_blocks[i], t):
            if cover_count.get(ts, 0) <= 1:
                is_redundant = False
                break
        if is_redundant:
            for ts in itertools.combinations(current_blocks[i], t):
                cover_count[ts] -= 1
            current_blocks.pop(i)
            current_count -= 1
            best_blocks = [list(b) for b in current_blocks]
            best_count = current_count
    
    # Try swapping points in blocks to create more redundancy
    # Key optimization: use cover_count to check validity in O(C(k,t)) per swap
    max_swap_iters = 500
    for _ in range(max_swap_iters):
        if time.time() > deadline:
            break
        
        if not current_blocks:
            break
        
        block_idx = rng.randint(0, len(current_blocks) - 1)
        block = current_blocks[block_idx]
        block_set = set(block)
        
        # Try swapping each point in the block with a random point not in the block
        found_swap = False
        for pos in range(len(block)):
            if time.time() > deadline:
                break
            if found_swap:
                break
            
            orig_point = block[pos]
            # Try a few random swaps
            for _ in range(8):
                new_point = rng.randint(0, v - 1)
                if new_point in block_set:
                    continue
                
                # Compute t-subsets that are LOST (contain orig_point, not in new_block)
                # and t-subsets that are GAINED (contain new_point)
                # A swap is valid iff every lost t-subset is still covered by another block
                # i.e., cover_count[lost_ts] >= 2 for all lost_ts
                
                # Lost t-subsets: those containing orig_point from the old block
                lost_tsubs = []
                for ts in itertools.combinations(block, t):
                    if orig_point in ts:
                        lost_tsubs.append(ts)
                
                valid = True
                for ts in lost_tsubs:
                    if cover_count.get(ts, 0) <= 1:
                        valid = False
                        break
                
                if valid:
                    # Accept the swap: update cover_count
                    for ts in lost_tsubs:
                        cover_count[ts] -= 1
                    
                    # Gained t-subsets: those containing new_point in the new block
                    new_block = block.copy()
                    new_block[pos] = new_point
                    new_block.sort()
                    new_block_set = set(new_block)
                    
                    gained_tsubs = []
                    for ts in itertools.combinations(new_block, t):
                        if new_point in ts:
                            gained_tsubs.append(ts)
                    
                    for ts in gained_tsubs:
                        cover_count[ts] = cover_count.get(ts, 0) + 1
                    
                    current_blocks[block_idx] = new_block
                    found_swap = True
                    break
    
    # After swaps, try to remove redundant blocks again
    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
        is_redundant = True
        for ts in itertools.combinations(current_blocks[i], t):
            if cover_count.get(ts, 0) <= 1:
                is_redundant = False
                break
        if is_redundant:
            for ts in itertools.combinations(current_blocks[i], t):
                cover_count[ts] -= 1
            current_blocks.pop(i)
            current_count -= 1
            best_blocks = [list(b) for b in current_blocks]
            best_count = current_count
    
    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 80
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Use a slightly tighter deadline for greedy to leave time for local search
        greedy_deadline = min(deadline, time.time() + (deadline - time.time()) * 0.6)
        blocks = _greedy_cover(v, k, t, rng, greedy_deadline)
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
