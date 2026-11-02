"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict
import heapq


def construct(v, k, t):
    deadline = time.time() + 145  # 145s deadline for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Greedy covering with dynamic scoring
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time_left() > 5:
        # Pick the most constrained uncovered t-subset
        # For efficiency, pick randomly but try multiple candidates
        candidates = list(uncovered)
        random.shuffle(candidates)
        num_to_try = min(15, len(candidates))
        
        best_block = None
        best_gain = -1
        
        for start_tsub in candidates[:num_to_try]:
            block = list(start_tsub)
            while len(block) < k and time_left() > 3:
                # Score each candidate point by how many uncovered t-subsets it would cover
                def gain(x):
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        tsub = tuple(sorted(rest + (x,)))
                        if tsub in uncovered:
                            count += 1
                    return count
                
                best_x = max((x for x in range(v) if x not in block), key=gain)
                block.append(best_x)
            block.sort()
            
            # Calculate gain of this block
            gain = sum(1 for tsub in itertools.combinations(block, t) if tsub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(best_block)
        uncovered.difference_update(itertools.combinations(best_block, t))
    
    if not uncovered:
        # Try local search to reduce block count
        return _local_search(v, k, t, blocks, deadline)
    
    return blocks


def _local_search(v, k, t, blocks, deadline):
    """Try to reduce block count via local search."""
    if time.time() > deadline - 5:
        return blocks
    
    # Build coverage map: for each t-subset, which blocks cover it
    cover_map = defaultdict(list)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            cover_map[tsub].append(i)
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # Try removing redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Check if block i is redundant (all its t-subsets covered by other blocks)
            redundant = True
            for tsub in itertools.combinations(best_blocks[i], t):
                if len(cover_map[tsub]) <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                removed_block = best_blocks[i]
                del best_blocks[i]
                # Update cover_map
                for tsub in itertools.combinations(removed_block, t):
                    cover_map[tsub].remove(i)
                # Reindex
                for tsub in cover_map:
                    cover_map[tsub] = [j if j < i else j - 1 for j in cover_map[tsub]]
                best_count = len(best_blocks)
                changed = True
                break  # Restart loop since indices changed
    
    # Try swapping blocks: replace a block with a better one that allows removing another
    random.seed(42)
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Pick a random block to try replacing
        idx = random.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]
        
        # Find t-subsets only covered by this block
        unique_tsubs = []
        for tsub in itertools.combinations(old_block, t):
            if len(cover_map[tsub]) == 1:
                unique_tsubs.append(tsub)
        
        if not unique_tsubs:
            # Block is redundant, remove it
            removed_block = best_blocks[idx]
            del best_blocks[idx]
            for tsub in itertools.combinations(removed_block, t):
                cover_map[tsub].remove(idx)
            for tsub in cover_map:
                cover_map[tsub] = [j if j < idx else j - 1 for j in cover_map[tsub]]
            best_count = len(best_blocks)
            continue
        
        # Try to find a new block that covers all unique_tsubs
        # The new block must contain all points that are in any unique_tsub
        union_size = len(set(pt for tsub in unique_tsubs for pt in tsub))
        if union_size > k:
            continue
        
        # The new block must contain the union of all unique_tsubs
        candidate = set(pt for tsub in unique_tsubs for pt in tsub)
        
        # Fill up to k elements
        while len(candidate) < k:
            # Add a point that covers the most uncovered t-subsets
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in candidate:
                    continue
                gain = 0
                for tsub in unique_tsubs:
                    if x not in tsub:
                        continue
                    # Check if adding x would help cover this tsub
                    if all(pt in candidate for pt in tsub if pt != x):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            candidate.add(best_point)
        
        if len(candidate) != k:
            continue
        
        new_block = sorted(candidate)
        
        # Verify new block covers all unique_tsubs
        valid = True
        for tsub in unique_tsubs:
            if not all(pt in new_block for pt in tsub):
                valid = False
                break
        
        if not valid:
            continue
        
        # Accept the replacement
        old_block_set = set(old_block)
        new_block_set = set(new_block)
        
        # Update cover_map: remove old block's contributions, add new block's
        for tsub in itertools.combinations(old_block, t):
            if idx in cover_map[tsub]:
                cover_map[tsub].remove(idx)
        
        for tsub in itertools.combinations(new_block, t):
            if tsub not in cover_map:
                cover_map[tsub] = []
            cover_map[tsub].append(idx)
        
        best_blocks[idx] = new_block
    
    return best_blocks
# EVOLVE-BLOCK-END
