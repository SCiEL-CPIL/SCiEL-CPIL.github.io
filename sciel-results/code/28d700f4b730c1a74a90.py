"""Improved program for C(19,7,4) covering design.
Uses efficient greedy construction with precomputed data structures,
followed by local search to reduce block count."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 115  # safety margin under 120s limit
    
    def time_left():
        return deadline - time.time()
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # Greedy covering with time budget
    uncovered = set(range(n_t))  # indices of uncovered t-subsets
    blocks = []
    
    while uncovered and time_left() > 10:
        best_block = None
        best_gain = -1
        
        # Try several random starting t-subsets
        candidates = list(uncovered)
        random.shuffle(candidates)
        num_to_try = min(30, len(candidates))
        
        for start_idx in candidates[:num_to_try]:
            block = list(all_t_subs[start_idx])
            while len(block) < k and time_left() > 5:
                # For each candidate point x, compute how many new t-subsets it covers
                best_x = -1
                best_x_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count t-subsets in (block + [x]) that are uncovered
                    x_gain = 0
                    for tsub in itertools.combinations(block + [x], t):
                        idx = t_sub_to_idx.get(tsub)
                        if idx is not None and idx in uncovered:
                            x_gain += 1
                    if x_gain > best_x_gain:
                        best_x_gain = x_gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            
            # Calculate total gain of this block
            gain = 0
            for tsub in itertools.combinations(block, t):
                idx = t_sub_to_idx.get(tsub)
                if idx is not None and idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(best_block)
        # Update uncovered
        for tsub in itertools.combinations(best_block, t):
            idx = t_sub_to_idx.get(tsub)
            if idx is not None:
                uncovered.discard(idx)
    
    if not uncovered:
        # Try local search to reduce block count
        return _local_search(v, k, t, blocks, all_t_subs, t_sub_to_idx, n_t, deadline)
    
    # Fallback: return what we have (may be invalid, but better than nothing)
    return blocks


def _local_search(v, k, t, blocks, all_t_subs, t_sub_to_idx, n_t, deadline):
    """Reduce block count via local search using cover_count for efficiency."""
    if time.time() > deadline - 5:
        return blocks
    
    best_blocks = list(blocks)
    n_blocks = len(best_blocks)
    
    # Build cover_count: for each t-subset index, how many blocks cover it
    cover_count = [0] * n_t
    block_tsub_indices = []  # for each block, list of t-subset indices it covers
    
    for i, block in enumerate(best_blocks):
        indices = []
        for tsub in itertools.combinations(block, t):
            idx = t_sub_to_idx.get(tsub)
            if idx is not None:
                indices.append(idx)
                cover_count[idx] += 1
        block_tsub_indices.append(indices)
    
    best_count = len(best_blocks)
    
    # Phase 1: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Check if block i is redundant
            redundant = True
            for idx in block_tsub_indices[i]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for idx in block_tsub_indices[i]:
                    cover_count[idx] -= 1
                del best_blocks[i]
                del block_tsub_indices[i]
                best_count = len(best_blocks)
                changed = True
                break  # Restart since indices changed
    
    # Phase 2: Try to replace blocks to enable further removals
    random.seed(42)
    attempts = 0
    max_attempts = 2000
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        if not best_blocks:
            break
        
        idx = random.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]
        
        # Find t-subsets only covered by this block
        unique_tsubs = []
        for tsub_idx in block_tsub_indices[idx]:
            if cover_count[tsub_idx] == 1:
                unique_tsubs.append(tsub_idx)
        
        if not unique_tsubs:
            # Block is redundant, remove it
            for tsub_idx in block_tsub_indices[idx]:
                cover_count[tsub_idx] -= 1
            del best_blocks[idx]
            del block_tsub_indices[idx]
            best_count = len(best_blocks)
            continue
        
        # Check if union of unique t-subsets fits in a k-set
        union_pts = set()
        for tsub_idx in unique_tsubs:
            union_pts.update(all_t_subs[tsub_idx])
        
        if len(union_pts) > k:
            continue
        
        # Try to find a new block that covers all unique t-subsets
        candidate = set(union_pts)
        
        # Fill up to k elements, prioritizing points that cover more unique t-subsets
        while len(candidate) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in candidate:
                    continue
                gain = 0
                for tsub_idx in unique_tsubs:
                    tsub = all_t_subs[tsub_idx]
                    if x not in tsub:
                        continue
                    if all(pt in candidate for pt in tsub if pt != x):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            candidate.add(best_point)
        
        if len(candidate) != k:
            continue
        
        new_block = sorted(candidate)
        
        # Verify new block covers all unique t-subsets
        valid = True
        new_block_set = set(new_block)
        for tsub_idx in unique_tsubs:
            tsub = all_t_subs[tsub_idx]
            if not all(pt in new_block_set for pt in tsub):
                valid = False
                break
        
        if not valid:
            continue
        
        # Accept the replacement
        # Update cover_count: remove old block's contributions, add new block's
        for tsub_idx in block_tsub_indices[idx]:
            cover_count[tsub_idx] -= 1
        
        new_indices = []
        for tsub in itertools.combinations(new_block, t):
            tsub_idx = t_sub_to_idx.get(tsub)
            if tsub_idx is not None:
                new_indices.append(tsub_idx)
                cover_count[tsub_idx] += 1
        
        best_blocks[idx] = new_block
        block_tsub_indices[idx] = new_indices
    
    return best_blocks


# EVOLVE-BLOCK-END
