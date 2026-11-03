"""Optimized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute(v, k, t):
    """Precompute all t-subsets and block->t-subset-index mappings."""
    # Map each t-subset (as a sorted tuple) to an index
    tsub_tuples = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(tsub_tuples)}
    total_t = len(tsub_tuples)
    
    # For each possible block (k-subset), precompute which t-subset indices it covers
    # We won't enumerate all C(24,7) blocks, but we'll compute on-the-fly with caching
    
    return tsub_tuples, tsub_to_idx, total_t


def _block_tsub_indices(block, tsub_to_idx, t):
    """Get the list of t-subset indices covered by a block."""
    indices = []
    for c in itertools.combinations(block, t):
        ts = tuple(sorted(c))
        idx = tsub_to_idx.get(ts)
        if idx is not None:
            indices.append(idx)
    return indices


def _greedy_cover(v, k, t, tsub_to_idx, total_t, rng):
    """Fast greedy construction."""
    uncovered = bytearray([1]) * total_t
    uncovered_count = total_t
    blocks = []
    seen_blocks = set()
    
    while uncovered_count > 0:
        # Find an uncovered t-subset
        while True:
            idx = rng.randrange(total_t)
            if uncovered[idx]:
                break
        
        # Build a block greedily starting from this subset
        seed = list(itertools.combinations(range(v), 1))  # placeholder
        # Actually, start with the uncovered subset's elements
        # We need to find which t-subset has index idx
        # Reconstruct: we stored tsub_tuples but not easily accessible by index here
        # Let's just pick random points and greedily add
        
        # Start with t random points (ensuring they form a valid start)
        block = []
        # Pick t random distinct points
        while len(block) < t:
            x = rng.randrange(v)
            if x not in block:
                block.append(x)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = 0
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many new t-subsets this would cover
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = tuple(sorted(list(rest) + [x]))
                    idx = tsub_to_idx.get(fs)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                # Add a random point to fill
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    block.append(rng.choice(remaining))
                break
            else:
                block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                ts = tuple(sorted(c))
                idx = tsub_to_idx.get(ts)
                if idx is not None and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1
    
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 145.0  # leave 5s buffer
    
    tsub_tuples, tsub_to_idx, total_t = _precompute(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, tsub_to_idx, total_t, rng)
    
    # counts[idx] = number of blocks covering t-subset idx
    counts = [0] * total_t
    
    def update_counts_add(block):
        for c in itertools.combinations(block, t):
            ts = tuple(sorted(c))
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                counts[idx] += 1
    
    def update_counts_remove(block):
        for c in itertools.combinations(block, t):
            ts = tuple(sorted(c))
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                counts[idx] -= 1
    
    for b in blocks:
        update_counts_add(b)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def is_redundant(block):
        """Check if removing block would leave all t-subsets covered."""
        for c in itertools.combinations(block, t):
            ts = tuple(sorted(c))
            idx = tsub_to_idx.get(ts)
            if idx is not None and counts[idx] <= 1:
                return False
        return True
    
    # Precompute block->t-subset-index lists for current blocks
    block_tsubs = []
    for b in blocks:
        block_tsubs.append(_block_tsub_indices(b, tsub_to_idx, t))
    
    def is_redundant_fast(block_tsub_list):
        for idx in block_tsub_list:
            if counts[idx] <= 1:
                return False
        return True
    
    def find_replacement(old_block, critical_indices):
        """Find a replacement block that covers all critical t-subsets."""
        if not critical_indices:
            return None
        
        # Points that must be in the new block
        points_needed = set()
        for idx in critical_indices:
            points_needed.update(tsub_tuples[idx])
        
        if len(points_needed) > k:
            return None
        
        new_block_set = set(points_needed)
        
        # Greedily fill remaining slots to maximize coverage of singly-covered subsets
        while len(new_block_set) < k:
            best_x = None
            best_gain = 0
            for x in range(v):
                if x in new_block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(list(new_block_set), t - 1):
                    fs = tuple(sorted(list(rest) + [x]))
                    idx = tsub_to_idx.get(fs)
                    if idx is not None:
                        c = counts[idx]
                        if c == 0:
                            gain += 1000
                        elif c == 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                # Just pick a random point
                remaining = [x for x in range(v) if x not in new_block_set]
                if not remaining:
                    break
                best_x = rng.choice(remaining)
            
            new_block_set.add(best_x)
        
        if len(new_block_set) < k:
            return None
        
        new_block = tuple(sorted(new_block_set))
        if new_block == old_block:
            return None
        
        # Validate: after removing old_block, all critical subsets must be covered by new_block
        new_block_tsubs = set(_block_tsub_indices(new_block, tsub_to_idx, t))
        for idx in critical_indices:
            if idx not in new_block_tsubs:
                return None
        
        return new_block
    
    # Main local search loop
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if is_redundant_fast(block_tsubs[i]):
                b = blocks[i]
                update_counts_remove(b)
                blocks.pop(i)
                block_tsubs.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    stuck_counter = 0
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Find critical t-subsets (count == 1)
            critical_indices = []
            for idx in block_tsubs[i]:
                if counts[idx] == 1:
                    critical_indices.append(idx)
            
            if critical_indices:
                new_block = find_replacement(old_block, critical_indices)
                if new_block is not None:
                    update_counts_remove(old_block)
                    new_tsubs = _block_tsub_indices(new_block, tsub_to_idx, t)
                    update_counts_add(new_block)
                    blocks[i] = new_block
                    block_tsubs[i] = new_tsubs
                    stuck_counter = 0
            else:
                # No critical subsets - try a random perturbation
                new_block_set = set(old_block)
                rem = rng.choice(list(new_block_set))
                new_block_set.remove(rem)
                cands = [x for x in range(v) if x not in new_block_set]
                if cands:
                    new_block_set.add(rng.choice(cands))
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        # Validate
                        new_block_tsubs = set(_block_tsub_indices(new_block, tsub_to_idx, t))
                        valid = True
                        for idx in block_tsubs[i]:
                            if counts[idx] == 1 and idx not in new_block_tsubs:
                                valid = False
                                break
                        
                        if valid:
                            update_counts_remove(old_block)
                            update_counts_add(new_block)
                            blocks[i] = new_block
                            block_tsubs[i] = list(new_block_tsubs)
                            stuck_counter = 0
        
        # 3. Restart if stuck
        stuck_counter += 1
        if stuck_counter > 300:
            new_cover = _greedy_cover(v, k, t, tsub_to_idx, total_t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                block_tsubs = [_block_tsub_indices(b, tsub_to_idx, t) for b in blocks]
                counts = [0] * total_t
                for b in blocks:
                    update_counts_add(b)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            stuck_counter = 0
    
    # Final pass: try to remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant_fast(block_tsubs[i]):
                b = blocks[i]
                update_counts_remove(b)
                blocks.pop(i)
                block_tsubs.pop(i)
                changed = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
