"""Covering design C(19,8,5): optimized randomized greedy with local search."""
import itertools
import random
import time
from collections import Counter


def _precompute_tsub_indices(v, k, t):
    """Precompute for each k-subset (as frozenset) the list of t-subset indices it covers."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    # Map each possible k-subset to the list of t-subset indices it contains
    # We'll compute on the fly but cache
    return all_tsubs, tsub_to_idx


def _greedy_pass(v, k, t, rng, time_limit, start_time, all_tsubs, tsub_to_idx):
    """One greedy pass with bitmask coverage tracking."""
    num_tsubs = len(all_tsubs)
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(num_tsubs))
    blocks = []
    
    # Precompute: for each point x and each (t-1)-subset of remaining points, 
    # which t-subset indices would be covered
    # Actually, let's compute gain more efficiently
    
    # For each k-subset, precompute which t-subset indices it covers
    # We'll build this lazily
    
    def block_tsub_indices(block):
        """Get the set of t-subset indices covered by a block."""
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    while uncovered:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick a random uncovered t-subset
        target_idx = rng.choice(list(uncovered))
        target = all_tsubs[target_idx]
        block = list(target)
        
        while len(block) < k:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            
            # For each candidate x not in block, compute gain
            # Gain = number of uncovered t-subsets that contain x and (t-1) elements from block
            # Precompute: for each x, and for each (t-1)-subset s of block, 
            # check if sorted(s + (x,)) is in uncovered
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms an uncovered t-subset
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(new_tsub)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_x = rng.choice(available)
                else:
                    break
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Remove all t-subsets of this block from uncovered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered.discard(idx)
    
    return blocks, uncovered


def _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time, all_tsubs, tsub_to_idx):
    """Iteratively remove blocks that don't cover any t-subset uniquely."""
    if not blocks:
        return blocks
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit - 3.0:
            break
        changed = False
        
        # Count how many blocks cover each t-subset
        tsub_counts = Counter()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tsub_counts[tsub_to_idx[ts]] += 1
        
        # Keep only blocks that cover at least one t-subset with count 1
        kept = []
        for block in blocks:
            if time.time() - start_time > time_limit - 3.0:
                kept.append(block)
                continue
            for ts in itertools.combinations(block, t):
                if tsub_counts[tsub_to_idx[ts]] == 1:
                    kept.append(block)
                    break
        
        if len(kept) < len(blocks):
            changed = True
            blocks = kept
    
    return blocks


def _local_search(v, k, t, blocks, time_limit, start_time, all_tsubs, tsub_to_idx, rng):
    """WalkSAT-style local search: try to remove or replace blocks."""
    if not blocks:
        return blocks
    
    num_tsubs = len(all_tsubs)
    
    def get_coverage(blocks):
        """Return set of covered t-subset indices."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(tsub_to_idx[ts])
        return covered
    
    def is_valid(blocks):
        if len(blocks) == 0:
            return False
        covered = get_coverage(blocks)
        return len(covered) == num_tsubs
    
    # Initial check
    if not is_valid(blocks):
        return blocks
    
    current = list(blocks)
    
    # Try removing each block one at a time
    while time.time() - start_time < time_limit - 5.0:
        improved = False
        n = len(current)
        order = list(range(n))
        rng.shuffle(order)
        
        for i in order:
            if time.time() - start_time > time_limit - 5.0:
                break
            # Try removing block i
            candidate = current[:i] + current[i+1:]
            if is_valid(candidate):
                current = candidate
                improved = True
                break
        
        if not improved:
            # Try replacing a block with a new one that covers more
            # Pick a random block to replace
            if n == 0:
                break
            i = rng.randrange(n)
            # Find a t-subset covered only by block i (if any)
            # Or just try to find a better block
            # For simplicity, try swapping: remove block i, add a new greedy block
            # This is expensive; only do a few iterations
            if time.time() - start_time > time_limit - 8.0:
                break
            # Skip complex replacement for now; just keep trying removals
            break
    
    return current


def construct(v, k, t):
    start_time = time.time()
    time_limit = 120.0
    
    # Precompute
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15.0:
            break
        
        rng = random.Random(42 + restart * 7)
        
        # Greedy pass
        blocks, remaining = _greedy_pass(v, k, t, rng, time_limit, start_time, all_tsubs, tsub_to_idx)
        
        if not blocks:
            continue
        
        # If we didn't finish, try to complete
        if remaining:
            elapsed = time.time() - start_time
            if elapsed < time_limit - 10.0:
                for idx in list(remaining):
                    if time.time() - start_time > time_limit - 8.0:
                        break
                    tsub = all_tsubs[idx]
                    block = list(tsub)
                    while len(block) < k:
                        if time.time() - start_time > time_limit - 8.0:
                            break
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in block:
                                continue
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                new_tsub = tuple(sorted(rest + (x,)))
                                j = tsub_to_idx.get(new_tsub)
                                if j is not None and j in remaining:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            available = [x for x in range(v) if x not in block]
                            if available:
                                best_x = rng.choice(available)
                            else:
                                break
                        block.append(best_x)
                    if len(block) == k:
                        block.sort()
                        blocks.append(tuple(block))
                        for ts in itertools.combinations(block, t):
                            remaining.discard(tsub_to_idx[ts])
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time, all_tsubs, tsub_to_idx)
        
        # Local search
        if time.time() - start_time < time_limit - 10.0:
            blocks = _local_search(v, k, t, blocks, time_limit, start_time, all_tsubs, tsub_to_idx, rng)
        
        # Verify coverage
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(tsub_to_idx[ts])
        
        if len(covered) == num_tsubs and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks, _ = _greedy_pass(v, k, t, rng, time_limit, start_time, all_tsubs, tsub_to_idx)
        # Remove redundant
        best_blocks = _remove_redundant_blocks(v, k, t, best_blocks, time_limit, start_time, all_tsubs, tsub_to_idx)
    
    return [list(b) for b in best_blocks]
