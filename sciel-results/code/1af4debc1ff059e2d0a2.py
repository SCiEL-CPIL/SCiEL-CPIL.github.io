"""Rewritten program: bitmask-based greedy construction + local search (swap/remove/add)."""
import itertools
import random
import time
from math import comb


def _tsub_bitmasks(v, t):
    """Return list of integer bitmasks for all t-subsets of {0..v-1}."""
    return [0]  # placeholder
    # We'll compute on the fly or precompute for small v
    # For v=19, t=4, C(19,4)=3876 which is manageable
    result = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        result.append(mask)
    return result


def _block_covers(block, v, t):
    """Return set of t-subset bitmasks covered by a given block (list of points)."""
    covers = []
    for combo in itertools.combinations(block, t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        covers.append(mask)
    return covers


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Precompute all t-subsets as bitmasks
    all_tsubs = _tsub_bitmasks(v, t)
    n_tsubs = len(all_tsubs)

    # Map each t-subset bitmask to its index
    tsub_index = {mask: i for i, mask in enumerate(all_tsubs)}

    # Precompute for each possible point x and each t-subset mask, whether x is in the subset
    # Actually, let's think differently.
    
    # Represent a block as a frozenset of points.
    # A block covers a t-subset if the t-subset is a subset of the block.
    
    # Greedy construction using bitmasks for uncovered tracking
    # uncovered[i] = True if t-subset i is not yet covered
    uncovered = bytearray([1] * n_tsubs)
    uncovered_count = n_tsubs

    blocks = []  # list of frozensets

    # For greedy: for each candidate block, compute how many uncovered t-subs it covers.
    # We build blocks greedily: start with a t-subset, then add points that maximize coverage.

    def _covered_count_for_block(block_set, uncovered_arr):
        """Count how many currently-uncovered t-subsets are covered by block_set."""
        count = 0
        for combo in itertools.combinations(block_set, t):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            idx = tsub_index.get(mask)
            if idx is not None and uncovered_arr[idx]:
                count += 1
        return count

    def _make_block(start_combo):
        """Build a block starting from start_combo (a tuple of t points)."""
        block = list(start_combo)
        block_set = frozenset(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                new_block_set = block_set | {x}
                gain = _covered_count_for_block(new_block_set, uncovered) - _covered_count_for_block(block_set, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                break
            block.append(best_x)
            block_set = frozenset(block)
        
        return frozenset(block)

    # Greedy phase
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find first uncovered t-subset
        first_uncovered_idx = -1
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break
        
        # Get the points of this t-subset
        # We need to recover the actual points from the bitmask
        mask = all_tsubs[first_uncovered_idx]
        start_points = []
        for p in range(v):
            if mask & (1 << p):
                start_points.append(p)
        
        new_block = _make_block(tuple(start_points))
        blocks.append(new_block)
        
        # Update uncovered
        for combo in itertools.combinations(new_block, t):
            cmask = 0
            for x in combo:
                cmask |= (1 << x)
            idx = tsub_index.get(cmask)
            if idx is not None and uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1
    
    if uncovered_count == 0:
        return [list(b) for b in blocks]
    
    # Local search phase: try to improve by swapping, removing, or adding blocks
    # We'll use a time-bounded local search
    
    def _is_valid(blocks_list):
        """Check if blocks_list covers all t-subsets."""
        covered = bytearray([0] * n_tsubs)
        for b in blocks_list:
            for combo in itertools.combinations(b, t):
                cmask = 0
                for x in combo:
                    cmask |= (1 << x)
                idx = tsub_index.get(cmask)
                if idx is not None:
                    covered[idx] = 1
        for i in range(n_tsubs):
            if not covered[i]:
                return False
        return True

    def _count_uncovered(blocks_list):
        """Count uncovered t-subsets."""
        covered = bytearray([0] * n_tsubs)
        for b in blocks_list:
            for combo in itertools.combinations(b, t):
                cmask = 0
                for x in combo:
                    cmask |= (1 << x)
                idx = tsub_index.get(cmask)
                if idx is not None:
                    covered[idx] = 1
        return sum(1 for i in range(n_tsubs) if not covered[i])

    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # Local search: try removing blocks, then re-adding if needed
    # Strategy: repeatedly try to remove a block; if still valid, keep removal.
    # Then try to add a new block that covers the most uncovered subsets.
    
    random.seed(42)
    
    def _local_search(blocks_list, time_limit):
        """Perform local search starting from blocks_list."""
        current = list(blocks_list)
        best = list(current)
        best_len = len(current)
        
        t_end = time.time() + time_limit
        
        # Phase 1: Try to remove blocks
        for i in range(len(current) - 1, -1, -1):
            if time.time() > t_end:
                break
            test = current[:i] + current[i+1:]
            if _is_valid(test):
                current = test
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        # Phase 2: Try swaps - replace one block with another
        # Generate candidate blocks from uncovered or random
        for _ in range(50):
            if time.time() > t_end:
                break
            
            # Try removing a random block and finding a replacement
            idx = random.randint(0, len(current) - 1)
            without = current[:idx] + current[idx+1:]
            
            # What's uncovered without this block?
            uncov = bytearray([1] * n_tsubs)
            for b in without:
                for combo in itertools.combinations(b, t):
                    cmask = 0
                    for x in combo:
                        cmask |= (1 << x)
                    idx2 = tsub_index.get(cmask)
                    if idx2 is not None:
                        uncov[idx2] = 0
            
            uncov_count = sum(uncov)
            if uncov_count == 0:
                # We can remove this block
                current = without
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                continue
            
            # Find a block that covers the most uncovered subsets
            best_new_block = None
            best_gain = 0
            # Try blocks that include points from the removed block
            removed_block = current[idx] if idx < len(current) else None
            # Actually we removed it, so let's try generating new blocks
            # Try all blocks that are supersets of some t-subset of the removed block
            if removed_block:
                for combo in itertools.combinations(removed_block, t):
                    candidate = list(combo)
                    # Extend to size k
                    remaining = [x for x in range(v) if x not in candidate]
                    random.shuffle(remaining)
                    for x in remaining[:k - t]:
                        candidate.append(x)
                    if len(candidate) == k:
                        gain = 0
                        for combo2 in itertools.combinations(candidate, t):
                            cmask = 0
                            for xx in combo2:
                                cmask |= (1 << xx)
                            idx3 = tsub_index.get(cmask)
                            if idx3 is not None and uncov[idx3]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = frozenset(candidate)
            
            if best_new_block and best_gain >= uncov_count:
                current = without + [best_new_block]
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        return best

    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_blocks = _local_search(best_blocks, remaining_time - 2)
    
    # Final validation
    if not _is_valid(best_blocks):
        # Fallback: return the original greedy blocks
        best_blocks = list(blocks)
        if not _is_valid(best_blocks):
            # Should not happen, but just in case
            pass
    
    return [list(b) for b in best_blocks]
