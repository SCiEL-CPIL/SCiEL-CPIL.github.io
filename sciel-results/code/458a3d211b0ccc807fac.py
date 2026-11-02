"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict
import heapq


def construct(v, k, t):
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    # Greedy covering with efficient data structures
    # For each point, track how many uncovered t-subsets it appears in
    # We'll use a priority queue to select the best block
    
    while uncovered and time_left() > 2:
        # Find the best block to add
        # A block covers C(k,t) t-subsets; we want to maximize newly covered
        best_block = None
        best_gain = -1
        
        # Sample candidate blocks to find a good one quickly
        # First, find a t-subset that's uncovered and build around it
        if uncovered:
            # Pick a random uncovered t-subset to start
            start_tsub = random.choice(list(uncovered))
            
            # Build a block around this t-subset
            candidate = set(start_tsub)
            while len(candidate) < k:
                # Find the point that covers the most uncovered t-subsets
                best_point = None
                best_point_gain = -1
                
                for x in range(v):
                    if x in candidate:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(candidate, t - 1):
                        tsub = tuple(sorted(rest + (x,)))
                        if tsub in uncovered:
                            gain += 1
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                        if best_point_gain == 0:
                            break
                
                if best_point is None or best_point_gain == 0:
                    # Add a random point
                    available = [x for x in range(v) if x not in candidate]
                    if available:
                        best_point = random.choice(available)
                    else:
                        break
                candidate.add(best_point)
            
            if len(candidate) == k:
                block = tuple(sorted(candidate))
                # Count gain
                gain = sum(1 for tsub in itertools.combinations(block, t) if tsub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
        
        if best_block is None or best_gain <= 0:
            # Fallback: pick any block
            if uncovered:
                start_tsub = random.choice(list(uncovered))
                candidate = set(start_tsub)
                while len(candidate) < k:
                    available = [x for x in range(v) if x not in candidate]
                    if available:
                        candidate.add(random.choice(available))
                    else:
                        break
                if len(candidate) == k:
                    best_block = tuple(sorted(candidate))
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)
    
    if not uncovered:
        # Try to reduce block count
        blocks = _reduce_blocks(v, k, t, blocks, deadline)
    
    return blocks


def _reduce_blocks(v, k, t, blocks, deadline):
    """Try to reduce block count by removing redundant blocks."""
    if time.time() > deadline - 5:
        return blocks
    
    # Build coverage count for each t-subset
    cover_count = defaultdict(int)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            cover_count[tsub] += 1
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Check if block i is redundant
            redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                if cover_count[tsub] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for tsub in itertools.combinations(blocks[i], t):
                    cover_count[tsub] -= 1
                del blocks[i]
                changed = True
    
    return blocks
# EVOLVE-BLOCK-END
