# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _tsub_index(t, v):
    """Return an array mapping each t-subset (as a tuple) to a unique integer index."""
    from math import comb
    n = comb(v, t)
    idx = np.empty(n, dtype=np.int64)
    i = 0
    for c in itertools.combinations(range(v), t):
        idx[i] = i
        i += 1
    # Build a dict from frozenset to index
    tsub_to_idx = {}
    i = 0
    for c in itertools.combinations(range(v), t):
        tsub_to_idx[frozenset(c)] = i
        i += 1
    return idx, tsub_to_idx, n

def _block_mask(block, t, tsub_to_idx):
    """Return a boolean mask of t-subsets covered by a block."""
    n = len(tsub_to_idx)
    mask = np.zeros(n, dtype=bool)
    for c in itertools.combinations(block, t):
        mask[tsub_to_idx[frozenset(c)]] = True
    return mask

def _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs):
    """Greedy covering with vectorized gain computation."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    # Precompute for each point x, the t-subsets that contain x
    # point_tsubs[x] = list of indices of t-subsets containing x
    point_tsubs = []
    for x in range(v):
        idxs = [i for i, fs in enumerate(tsub_to_idx) if x in fs]
        point_tsubs.append(np.array(idxs, dtype=np.int64))
    
    # Also precompute t-1 subsets for each point: for gain when adding x to block B,
    # we need t-subsets {x} + (t-1 from B). Precompute all (t-1)-subsets of each point set? 
    # That's expensive. Instead, compute gain on the fly using the block's (t-1) subsets.
    
    while uncovered.any():
        if deadline and time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered)[0]
        seed_idx = rng.choice(uncovered_indices)
        
        # Find the actual t-subset from seed_idx
        # We need to reverse the index -> subset mapping
        # Precompute reverse mapping
        pass
        
        # Start block with that t-subset
        # We need to know which subset seed_idx corresponds to
        # Let's precompute a list of all t-subsets
        pass
        
        break  # placeholder
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s out of 150s, leaving 35s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}
    idx_to_tsub = all_tsubs  # index -> tuple
    
    # Precompute for each point x, the set of t-subset indices that contain x
    point_tsubs = []
    for x in range(v):
        idxs = []
        for i, c in enumerate(all_tsubs):
            if x in c:
                idxs.append(i)
        point_tsubs.append(np.array(idxs, dtype=np.int64))
    
    # Precompute (t-1) subsets for each point as a list of (t-1)-tuples
    # For gain computation: adding x to block B covers t-subsets {x} + S where S is (t-1)-subset of B
    # We need to map each (t-1)-subset S to the index of {x}+S
    # Precompute: for each x, a dict mapping frozenset((t-1)-subset) -> index of {x}+subset
    point_tsub_lookup = []
    for x in range(v):
        lookup = {}
        for i, c in enumerate(all_tsubs):
            if x in c:
                rest = tuple(sorted(set(c) - {x}))
                lookup[frozenset(rest)] = i
        point_tsubs_lookup = lookup
        point_tsub_lookup.append(lookup)
    
    def _greedy(v, k, t, rng, deadline, uncovered_init=None):
        """Greedy covering with vectorized gain."""
        if uncovered_init is not None:
            uncovered = uncovered_init.copy()
        else:
            uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while uncovered.any():
            if deadline and time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            seed_idx = rng.choice(uncovered_indices)
            seed_tsub = idx_to_tsub[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)
            
            # Precompute (t-1) subsets of current block
            t_minus_1 = t - 1
            
            while len(block) < k:
                if deadline and time.time() > deadline:
                    break
                
                t1_subsets = list(itertools.combinations(block, t_minus_1))
                
                # For each candidate point x not in block, compute gain
                best_gain = -1
                best_point = None
                
                # Build a numpy array of gains for all candidate points
                gains = np.zeros(v, dtype=np.int64)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    lookup = point_tsub_lookup[x]
                    for rest in t1_subsets:
                        key = frozenset(rest)
                        if key in lookup:
                            idx = lookup[key]
                            if uncovered[idx]:
                                gain += 1
                    gains[x] = gain
                
                # Find best point
                mask = np.ones(v, dtype=bool)
                for x in block:
                    mask[x] = False
                valid_gains = gains[mask]
                if len(valid_gains) > 0:
                    max_gain = np.max(valid_gains)
                    if max_gain > best_gain:
                        # Find all points with max gain, pick one at random
                        candidates = np.where(mask & (gains == max_gain))[0]
                        best_point = int(rng.choice(candidates))
                        best_gain = max_gain
                
                if best_point is None:
                    # Fallback
                    for x in range(v):
                        if x not in block_set:
                            best_point = x
                            break
                
                if best_point is not None:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            
            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                # Update uncovered
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered[idx] = False
        
        return [list(b) for b in blocks]
    
    def _local_search(blocks, deadline):
        """Remove redundant blocks and try replacements."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        
        # Compute coverage count for each t-subset
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[frozenset(c)]] = True
            block_masks.append(mask)
            cov_count += mask
        
        # Phase 1: Remove redundant blocks (those where all covered t-subsets have count > 1)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if deadline and time.time() > deadline:
                    return blocks
                # Check if block i is redundant
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    # All t-subsets covered by block i are also covered by others
                    cov_count -= mask
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break
        
        # Phase 2: Try replacing each block with a better one
        for i in range(len(blocks)):
            if deadline and time.time() > deadline:
                break
            # Find t-subsets only covered by block i
            mask_i = block_masks[i]
            unique = mask_i & (cov_count == 1)
            if not unique.any():
                continue
            
            # Try to find a block that covers all unique t-subsets
            unique_indices = np.where(unique)[0]
            # For each unique t-subset, find blocks that cover it
            # A replacement block must cover all unique t-subsets
            # This is expensive, so limit the search
            
            # Simple heuristic: try a few random blocks
            for _ in range(20):
                if deadline and time.time() > deadline:
                    break
                new_block = [rng_sample := random.Random(i * 7 + _).choice(range(v)) for _ in range(k)]
                # Deduplicate
                new_block = list(set(new_block))
                if len(new_block) < k:
                    continue
                new_block.sort()
                new_mask = np.zeros(n_tsubs, dtype=bool)
                for c in itertools.combinations(new_block, t):
                    new_mask[tsub_to_idx[frozenset(c)]] = True
                if np.all(new_mask[unique_indices]):
                    # Replace
                    cov_count -= mask_i
                    cov_count += new_mask
                    blocks[i] = new_block
                    block_masks[i] = new_mask
                    break
        
        return blocks
    
    # Run greedy with multiple restarts
    best_blocks = None
    best_len = float('inf')
    
    # Baseline deterministic greedy
    rng0 = random.Random(42)
    baseline = _greedy(v, k, t, rng0, deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)
    
    # Random restarts
    restart = 0
    while time.time() < deadline - 40:  # Keep 40s for local search
        restart += 1
        rng = random.Random(restart * 12345 + 7)
        blocks = _greedy(v, k, t, rng, deadline)
        
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Local search on best
    if best_blocks and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, deadline)
    
    return best_blocks if best_blocks else []

# EVOLVE-BLOCK-END
