"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_init(v, k, t, rng):
    """
    Initialize a valid covering using a greedy set cover approach.
    Returns a list of blocks (tuples of sorted ints).
    """
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    # Precompute all possible blocks? No, generate on fly.
    # For v=24, k=7, C(24,7) is large, but we only add blocks when needed.
    
    while uncovered:
        # Pick a random uncovered 3-subset to start a new block
        seed = rng.choice(list(uncovered))
        block_set = set(seed)
        
        # Greedily expand the block to size k to maximize coverage of uncovered subsets
        while len(block_set) < k:
            best_point = None
            max_gain = -1
            candidates = [x for x in range(v) if x not in block_set]
            rng.shuffle(candidates)
            
            for x in candidates:
                # Calculate gain: how many new uncovered 3-subsets are formed
                gain = 0
                # A new 3-subset containing x is formed by x and 2 points from block_set
                for pair in itertools.combinations(sorted(block_set), t - 1):
                    sub = frozenset(list(pair) + [x])
                    if sub in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block_set.add(best_point)
        
        # Sort and add block
        block_tuple = tuple(sorted(block_set))
        if block_tuple not in blocks:
            blocks.append(block_tuple)
            # Update uncovered set
            for sub in itertools.combinations(block_tuple, t):
                uncovered.discard(frozenset(sub))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # 1. Initialization
    blocks = _greedy_init(v, k, t, rng)
    best_blocks = list(blocks)
    best_size = len(blocks)
    
    # Precompute all 3-subsets as frozensets for quick lookup?
    # We maintain a count dictionary.
    
    def get_counts(blk_list):
        counts = {}
        for b in blk_list:
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = get_counts(blocks)
    
    def apply_change(old_block, new_block):
        """
        Replace old_block with new_block in the current state.
        Returns True if valid (no uncovered subsets created), False otherwise.
        Actually, we just update counts and check if any count drops to 0.
        """
        if old_block is not None:
            for sub in itertools.combinations(old_block, t):
                f = frozenset(sub)
                counts[f] -= 1
                if counts[f] == 0:
                    del counts[f]
        
        if new_block is not None:
            for sub in itertools.combinations(new_block, t):
                f = frozenset(sub)
                counts[f] = counts.get(f, 0) + 1
                
    def is_valid():
        # Check if any 3-subset is missing
        # We don't have a list of all 3-subsets stored separately for O(1) check,
        # but we know total is C(v, t).
        # If len(counts) == total_t, it's valid.
        return len(counts) == total_t

    total_t = len(list(itertools.combinations(range(v), t)))
    
    # 2. Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if all its 3-subsets have count > 1
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block
                apply_change(b, None)
                blocks.pop(i)
                improved = True
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = list(blocks)
                # Restart removal phase as counts changed
                i = 0
            else:
                i += 1
        
        # Phase 2: Swap blocks
        # Try to swap a block with another that covers its critical subsets
        # Critical subsets: those with count == 1 in the current block
        swapped = False
        for i in range(len(blocks)):
            b = blocks[i]
            critical = []
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            if not critical:
                continue # Should have been removed in Phase 1
            
            # Find points that must be in the new block
            # Union of all critical subsets
            points_needed = set()
            for f in critical:
                points_needed.update(f)
            
            if len(points_needed) > k:
                # Cannot cover all critical subsets with a single block
                continue
            
            # Try to form a new block containing points_needed
            # Fill the rest randomly
            remaining_points = [x for x in range(v) if x not in points_needed]
            rng.shuffle(remaining_points)
            
            # Try a few random completions
            for _ in range(10):
                new_block_set = set(points_needed)
                for x in remaining_points[:k - len(points_needed)]:
                    new_block_set.add(x)
                
                new_block = tuple(sorted(new_block_set))
                
                if new_block == b:
                    continue
                
                # Check if new_block covers all critical subsets
                # Since points_needed is union of critical, and new_block contains points_needed,
                # does it cover all critical subsets?
                # Yes, if for every critical subset C, C is a subset of new_block.
                # Since C is subset of points_needed, and points_needed is subset of new_block, yes.
                
                # Check validity of swap
                # 1. Remove old block
                # 2. Add new block
                # 3. Check if any count drops to 0?
                #    Critical subsets had count 1. They are covered by new_block, so count becomes 1 (from new) + 0 (from old removal? No, old removal drops to 0, new add brings to 1).
                #    Non-critical subsets had count >= 2. Old removal drops to >= 1. New add might increase.
                #    So validity is guaranteed IF new_block covers all critical subsets.
                
                # However, we must ensure new_block is distinct from existing blocks?
                # The problem says "blocks must be distinct".
                # If new_block is already in blocks, we can't add it (would create duplicate).
                # But if we remove b and add new_block, and new_block is already there, we just have one less copy of b?
                # No, blocks is a list of distinct blocks.
                # If new_block is in blocks, we can't swap b -> new_block because we'd have duplicate new_block?
                # Actually, if new_block is already in the list, we can just remove b.
                # But then we don't add anything.
                # So:
                if new_block in blocks:
                    # Just remove b
                    apply_change(b, None)
                    blocks.pop(i)
                    improved = True
                    swapped = True
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = list(blocks)
                    break
                else:
                    # Perform swap
                    apply_change(b, new_block)
                    blocks[i] = new_block
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                # Restart loop as state changed
                break
        
        if improved:
            continue
            
        # Phase 3: Perturbation (Kick)
        # If stuck, remove a random block and cover the holes
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            b = blocks[i]
            
            # Save state
            old_blocks = list(blocks)
            old_counts = dict(counts)
            
            # Remove block
            apply_change(b, None)
            blocks.pop(i)
            
            # Find uncovered subsets
            uncovered = []
            # We need to check which 3-subsets are now missing
            # We don't have a list of ALL 3-subsets readily available for iteration without generating
            # But we can generate them. v=24, C(24,3)=2024.
            for sub in itertools.combinations(range(v), t):
                f = frozenset(sub)
                if counts.get(f, 0) == 0:
                    uncovered.append(f)
            
            # Greedily cover uncovered subsets
            new_blocks_added = 0
            while uncovered:
                # Pick a random uncovered subset
                seed = rng.choice(uncovered)
                block_set = set(seed)
                
                while len(block_set) < k:
                    best_point = None
                    max_gain = -1
                    candidates = [x for x in range(v) if x not in block_set]
                    rng.shuffle(candidates)
                    
                    for x in candidates:
                        gain = 0
                        for pair in itertools.combinations(sorted(block_set), t - 1):
                            sub = frozenset(list(pair) + [x])
                            if sub in uncovered:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_point = x
                    
                    if best_point is None:
                        break
                    block_set.add(best_point)
                
                new_block = tuple(sorted(block_set))
                
                # Check if new_block is already in blocks
                if new_block in blocks:
                    # It's already there, so it covers these subsets?
                    # If it's in blocks, its subsets are already counted.
                    # But we just removed b. If new_block was in blocks, its count wasn't affected by removing b.
                    # So if new_block is in blocks, the subsets it covers are still covered.
                    # So we don't need to add it.
                    # But wait, if it's in blocks, why were the subsets uncovered?
                    # They wouldn't be. So this case implies the subsets are covered by existing blocks.
                    # So we can just remove them from uncovered list.
                    for sub in itertools.combinations(new_block, t):
                        uncovered.remove(frozenset(sub))
                    continue
                
                # Add new block
                blocks.append(new_block)
                apply_change(None, new_block)
                new_blocks_added += 1
                
                # Update uncovered list
                # Remove any subsets covered by new_block
                new_uncovered = []
                for f in uncovered:
                    # Check if f is covered by new_block
                    # f is a frozenset of 3 points.
                    # If all 3 points are in new_block, it's covered.
                    if all(p in new_block for p in f):
                        pass # Covered
                    else:
                        new_uncovered.append(f)
                uncovered = new_uncovered
            
            # If we added more blocks than we removed, size increased.
            # If we added 1, size same.
            # If we added 0 (impossible if uncovered was not empty), size decreased.
            
            # If the perturbation didn't lead to an improvement immediately,
            # we might be in a worse state.
            # However, the next iteration's Phase 1 (Removal) will try to fix it.
            # If Phase 1 and 2 fail again, we might loop.
            # To avoid infinite loops of bad perturbations, we can revert if no improvement after a few kicks?
            # For simplicity, we just continue. The "improved" flag was False, so we entered Phase 3.
            # After Phase 3, we restart the main loop.
            
            # If the new size is worse than best, we still keep it to explore.
            # But if it's much worse, maybe revert?
            # Let's just let the local search work.
            
            # Mark as improved to restart loop?
            # Yes, state changed.
            improved = True

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
