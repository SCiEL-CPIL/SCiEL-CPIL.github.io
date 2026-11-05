import time
import itertools
import random
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Track coverage: for each t-subset, how many blocks cover it
    coverage = [0] * n_subsets
    
    # For each point, precompute the list of t-subsets that contain it
    point_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for pt in s:
            point_subsets[pt].append(i)
    
    # Helper: get list of uncovered t-subset indices
    def get_uncovered():
        return [i for i in range(n_subsets) if coverage[i] == 0]
    
    # Greedy block construction: given a seed subset, build a block of size k
    def build_block(seed, uncovered_set):
        if not uncovered_set:
            return None
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            best_pt = None
            best_gain = -1
            for pt in range(v):
                if pt in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered
                # A t-subset is covered if it contains the new point and t-1 points from the block
                gain = 0
                for idx in point_subsets[pt]:
                    if coverage[idx] > 0:
                        continue
                    s = all_subsets[idx]
                    # Check if s contains pt and t-1 elements from block
                    # s contains pt by construction (it's in point_subsets[pt])
                    # Check if the other t-1 elements are in block
                    count_in_block = 0
                    for elem in s:
                        if elem != pt and elem in block_set:
                            count_in_block += 1
                    if count_in_block == t - 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pt = pt
            if best_pt is None:
                break
            block.append(best_pt)
            block_set.add(best_pt)
        
        if len(block) < k:
            # Fill with random points
            remaining = [pt for pt in range(v) if pt not in block_set]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        return block
    
    # Evaluate a block: return list of t-subset indices it covers
    def block_covers(block):
        covered = set()
        for s in itertools.combinations(block, t):
            idx = subset_to_idx[tuple(s)]
            covered.add(idx)
        return list(covered)
    
    blocks = []
    block_set = set()
    
    # Phase 1: Pure greedy - always pick the block that covers the most uncovered subsets
    while True:
        if time.time() > deadline - 5:
            break
        uncovered = get_uncovered()
        if not uncovered:
            break
        
        # Try to find a good block by starting from different seeds
        best_block = None
        best_cover_count = 0
        best_covered = []
        
        # Try several seed strategies
        seeds_to_try = []
        
        # Strategy 1: Start from the first uncovered subset
        seeds_to_try.append(uncovered[0])
        
        # Strategy 2: Start from a random uncovered subset
        if len(uncovered) > 1:
            seeds_to_try.append(random.choice(uncovered))
        
        # Strategy 3: Start from a subset with many uncovered neighbors
        if len(uncovered) > 5:
            # Pick a few random ones
            for _ in range(min(5, len(uncovered))):
                seeds_to_try.append(random.choice(uncovered))
        
        for seed_idx in seeds_to_try:
            seed = all_subsets[seed_idx]
            block = build_block(seed, uncovered)
            if block is None:
                continue
            block_tuple = tuple(block)
            if block_tuple in block_set:
                continue
            covered = block_covers(block)
            # Count how many of these are currently uncovered
            cover_count = sum(1 for idx in covered if coverage[idx] == 0)
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_block = block
                best_covered = covered
                best_block_tuple = block_tuple
        
        if best_block is None or best_cover_count == 0:
            # Fallback: just add a random block
            block = list(random.sample(range(v), k))
            block.sort()
            best_block = block
            best_block_tuple = tuple(block)
            best_covered = block_covers(block)
        
        # Add the block
        blocks.append(best_block)
        block_set.add(best_block_tuple)
        for idx in best_covered:
            coverage[idx] += 1
    
    # Check if we're done
    if all(c > 0 for c in coverage):
        return blocks
    
    # Phase 2: Local search - try to improve by swapping blocks
    # This is a simplified version; we try to replace blocks that cover few new subsets
    if time.time() > deadline - 2:
        return blocks
    
    # Try to remove redundant blocks
    for i in range(len(blocks) - 1, -1, -1):
        if time.time() > deadline - 2:
            break
        # Check if removing block i still leaves all subsets covered
        # Temporarily remove
        for idx in block_covers(blocks[i]):
            coverage[idx] -= 1
        if all(c > 0 for c in coverage):
            blocks.pop(i)
            block_set.discard(tuple(blocks[i]))
            # Don't restore coverage since we removed the block
        else:
            # Restore
            for idx in block_covers(blocks[i]):
                coverage[idx] += 1
    
    return blocks
