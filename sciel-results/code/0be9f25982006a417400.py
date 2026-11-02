"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 145  # 145s deadline for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Greedy covering with time budget
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time_left() > 5:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            best_x = max((x for x in range(v) if x not in block), key=gain)
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
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
            del best_blocks[i]
            # Update cover_map
            for tsub in itertools.combinations(best_blocks[i], t):
                cover_map[tsub].remove(i)
            # Reindex
            for tsub in cover_map:
                cover_map[tsub] = [j if j < i else j - 1 for j in cover_map[tsub]]
            best_count = len(best_blocks)
    
    # Try swapping blocks: replace a block with a better one
    random.seed(42)
    attempts = 0
    max_attempts = 1000
    
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
            continue
        
        # Try to find a new block that covers all unique_tsubs
        # The new block must contain at least one point from each unique_tsub
        # and must be a valid k-subset
        
        # Greedily build a candidate block
        candidate = set()
        for tsub in unique_tsubs:
            # Add one random element from this tsub
            candidate.add(random.choice(tsub))
        
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
        
        # Check if this replacement reduces total block count or keeps it same
        # (we can't increase, and we prefer to keep same count for now)
        # Actually, this swap keeps count the same. To reduce count, we'd need
        # to remove a block entirely.
        
        # For now, just accept if it doesn't make things worse
        # (This is a placeholder for more sophisticated local search)
    
    return best_blocks
# EVOLVE-BLOCK-END
