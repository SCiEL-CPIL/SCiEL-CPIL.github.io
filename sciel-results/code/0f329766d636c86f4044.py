"""Optimized covering design C(19,8,5) with fast greedy + post-optimization."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets and assign indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets it participates in
    # point_to_tsubsets[x] = list of tsubset indices containing x
    point_to_tsubsets = [[] for _ in range(v)]
    for ts in all_tsubsets:
        for x in ts:
            point_to_tsubsets[x].append(tsubset_index[ts])
    
    # Track uncovered t-subsets as a set of indices
    uncovered = set(range(n_tsubsets))
    
    # For fast gain computation: for a partial block, compute which t-subsets
    # would be covered by adding point x
    # A t-subset is covered by block if all its elements are in block
    # For partial block B (size < k), adding x covers t-subsets that:
    #   - contain x
    #   - have all other (t-1) elements in B
    #   - are currently uncovered
    
    def compute_gain(block, x):
        """Count uncovered t-subsets that would be covered by adding x to block."""
        if len(block) < t - 1:
            return 0
        # t-subsets containing x with remaining (t-1) elements in block
        count = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = tsubset_index.get(ts)
            if idx is not None and idx in uncovered:
                count += 1
        return count
    
    def greedy_cover(uncovered_set, existing_blocks, max_blocks=None):
        """Continue greedy from current state."""
        blocks = list(existing_blocks)
        uncovered = set(uncovered_set)
        
        while uncovered:
            if time.time() > deadline:
                break
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
            
            # Find the t-subset with fewest covering blocks (most constrained)
            # Use the first uncovered as starting point, but try to pick a "hard" one
            # For speed, just pick min
            start_ts = min(uncovered)
            block = list(all_tsubsets[start_ts])
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                # Find best point to add
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x not in block:
                        g = compute_gain(block, x)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                # Remove covered t-subsets
                for ts in itertools.combinations(block, t):
                    uncovered.discard(tsubset_index[ts])
        
        return blocks, uncovered
    
    # Run greedy
    blocks, remaining_uncovered = greedy_cover(uncovered, [])
    
    if remaining_uncovered:
        # Try to cover remaining with additional blocks (shouldn't happen if greedy completed)
        while remaining_uncovered and time.time() < deadline:
            start_ts = min(remaining_uncovered)
            block = list(all_tsubsets[start_ts])
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x not in block:
                        g = compute_gain(block, x)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                for ts in itertools.combinations(block, t):
                    remaining_uncovered.discard(tsubset_index[ts])
    
    # Post-optimization: try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 10:
        # Build coverage count: for each t-subset, how many blocks cover it
        tsubset_block_count = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tsubset_block_count[tsubset_index[ts]] += 1
        
        # Try to remove blocks that are fully redundant
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if this block is redundant
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = tsubset_index[ts]
                    if tsubset_block_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        tsubset_block_count[tsubset_index[ts]] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
    
    # Convert to list of lists
    return [list(b) for b in blocks]
