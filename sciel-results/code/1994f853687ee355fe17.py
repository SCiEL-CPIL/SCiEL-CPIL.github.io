"""Covering design C(24,7,3) via randomized greedy + redundancy removal."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to be safe under 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each k-subset, which t-subsets it covers
    # This is expensive, so we'll compute on the fly but cache results
    # For v=24, k=7, t=3: C(24,7) = 346104 blocks, C(7,3) = 35 t-subsets per block
    
    best_blocks = None
    best_count = num_tsubsets  # worst case upper bound
    
    def greedy_cover(seed, time_limit):
        """Run a single greedy covering attempt with given random seed."""
        rng = random.Random(seed)
        uncovered = set(range(num_tsubsets))  # indices of uncovered t-subsets
        blocks = []
        
        while uncovered and time.time() < time_limit:
            # Pick a random uncovered t-subset to start the block
            start_ts = rng.choice(list(uncovered))
            start_combo = all_tsubsets[start_ts]
            block = list(start_combo)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_x = -1
                best_gain = -1
                # Try all points not in block
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(new_ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0:
                        # Tie-breaking: use random
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    # No valid point found (shouldn't happen)
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered.discard(idx)
        
        if not uncovered:
            return [list(b) for b in blocks]
        return None
    
    def remove_redundant(blocks, time_limit):
        """Try to remove redundant blocks from a covering."""
        if time.time() > time_limit:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        # Try to remove each block if it doesn't make any t-subset uncovered
        result = list(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result)):
                if time.time() > time_limit:
                    break
                block = result[i]
                # Check if removing this block would leave any t-subset uncovered
                can_remove = True
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove block i
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    result.pop(i)
                    changed = True
                    break  # Restart from beginning
        return [list(b) for b in result]
    
    # Try multiple randomized greedy attempts
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time for this attempt
        attempt_time = min(remaining - 2, 30)  # Leave 2s buffer
        
        blocks = greedy_cover(seed, time.time() + attempt_time)
        if blocks is not None:
            # Try to remove redundant blocks
            blocks = remove_redundant(blocks, time.time() + min(remaining - 2, 20))
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we're doing well, continue trying to improve
                if best_count <= 76:
                    # Good result, keep searching
                    pass
            seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(num_tsubsets))
        blocks = []
        while uncovered:
            start_ts = min(uncovered)
            start_combo = all_tsubsets[start_ts]
            block = list(start_combo)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(new_ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered.discard(idx)
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
