import itertools
import random
import time
import numpy as np
from collections import defaultdict

def _precompute_block_sigs(v, k, t):
    """
    Precompute bitset signatures for all k-subsets of v points.
    Returns:
    - tsub_to_idx: dict mapping t-subset tuple to index
    - block_to_tsubs: list of arrays of t-subset indices for each block
    - all_blocks: list of all k-subsets (tuples)
    - sigs: numpy array of shape (num_blocks, num_uint64s) representing bitsets
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    num_tsubs = len(all_tsubs)
    
    # Number of uint64s needed to represent num_tsubs bits
    num_uint64s = (num_tsubs + 63) // 64
    
    # Precompute t-subsets for each block
    # This is memory intensive if stored as list of lists, but we can compute on fly or store compactly
    # For v=24, k=7, t=3: 346,104 blocks. Each covers 35 t-subsets.
    # 346k * 35 ints is ~12M ints. Manageable.
    
    block_tsubs_indices = []
    for block in all_blocks:
        indices = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs_indices.append(indices)
        
    # Create bitset signatures
    sigs = np.zeros((num_blocks, num_uint64s), dtype=np.uint64)
    for i, indices in enumerate(block_tsubs_indices):
        for idx in indices:
            sigs[i, idx // 64] |= (1 << (idx % 64))
            
    return tsub_to_idx, block_tsubs_indices, all_blocks, sigs, num_tsubs, num_uint64s

def _greedy_cover_numpy(v, k, t, rng, deadline, tsub_to_idx, block_tsubs_indices, all_blocks, sigs, num_tsubs, num_uint64s):
    """
    Greedy covering using NumPy vectorized gain calculation.
    """
    # uncovered bitset
    uncovered = np.ones(num_uint64s, dtype=np.uint64)
    # Set all bits for num_tsubs
    if num_tsubs % 64 != 0:
        uncovered[-1] = (1 << (num_tsubs % 64)) - 1
    
    selected_blocks = []
    selected_indices = set()
    
    # Map from t-subset index to list of block indices covering it
    # This is needed for local search, but for greedy we just pick max gain.
    
    while True:
        if time.time() > deadline:
            break
        
        # Check if uncovered is zero
        if not np.any(uncovered):
            break
            
        # Calculate gain for all blocks
        # gain[i] = popcount(sigs[i] & uncovered)
        # We can do this by iterating over uint64 chunks?
        # Or just:
        # intersected = sigs & uncovered[None, :]
        # gains = np.count_nonzero(intersected, axis=1)
        # This creates a huge temporary array (346k * 32). 11M uint64s ~ 90MB. Might be okay.
        
        # To save memory/time, we can sample or use a heuristic.
        # But let's try the full vectorized approach first.
        
        # Optimization: Only consider blocks that cover at least one uncovered t-subset?
        # The gain calculation does this implicitly (gain=0 for others).
        
        # To avoid allocating huge intermediate arrays every step (which is slow),
        # we can pre-compute the intersection logic? No, uncovered changes.
        
        # Alternative: Iterate through uncovered t-subsets, pick one randomly, 
        # then find the best block covering it.
        # This is "randomized greedy".
        
        # Let's stick to full vectorized gain if it fits in memory.
        # 346104 * 32 uint64 = 11,075,328 uint64 = 88.6 MB.
        # NumPy operations on this are fast.
        
        intersected = sigs & uncovered[np.newaxis, :]
        # Count set bits in each row
        # np.bit_count is available in NumPy 1.24+?
        # If not, we can use a lookup table or loop.
        # Let's assume a standard environment. If bit_count is not available, we fallback.
        try:
            gains = np.bit_count(intersected, axis=1)
        except AttributeError:
            # Fallback for older NumPy
            # This will be slow, but let's hope for newer NumPy
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                gains[i] = sum(bin(x).count('1') for x in intersected[i])

        # Find max gain
        max_gain = np.max(gains)
        if max_gain == 0:
            # Should not happen if uncovered is not empty and we have all blocks
            break
            
        # Pick a random block among those with max gain
        candidates = np.where(gains == max_gain)[0]
        best_block_idx = rng.choice(candidates)
        
        selected_blocks.append(all_blocks[best_block_idx])
        selected_indices.add(best_block_idx)
        
        # Update uncovered
        uncovered &= ~sigs[best_block_idx]

    return selected_blocks

def _remove_redundant_fast(blocks, v, t, tsub_to_idx, block_tsubs_indices, deadline):
    """
    Remove redundant blocks.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    if not blocks:
        return blocks
        
    current_blocks = list(blocks)
    
    # Map block tuple to index in current_blocks?
    # Or just iterate.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Count coverage of each t-subset by current_blocks
        # We need to know if removing block i leaves any t-subset with count 0.
        
        # Efficient way:
        # For each block i, check if there exists a t-subset ts in block i
        # such that no other block j covers ts.
        
        # Precompute t-subsets for each block in current set
        # This is small (fewer than 100 blocks usually)
        
        # Let's use a counter dict for t-subsets
        # Key: t-subset tuple, Value: count
        tsub_counts = defaultdict(int)
        block_tsubs_list = []
        
        for block in current_blocks:
            tsubs = list(itertools.combinations(block, t))
            block_tsubs_list.append(tsubs)
            for ts in tsubs:
                tsub_counts[ts] += 1
                
        new_blocks = []
        for i, block in enumerate(current_blocks):
            is_redundant = True
            for ts in block_tsubs_list[i]:
                if tsub_counts[ts] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove it
                for ts in block_tsubs_list[i]:
                    tsub_counts[ts] -= 1
                changed = True
            else:
                new_blocks.append(block)
                
        if not new_blocks:
            return [] # Should not happen if valid
            
        current_blocks = new_blocks
        
    return current_blocks

def construct(v, k, t):
    deadline = time.time() + 145 # Safety margin
    
    # Precomputation
    tsub_to_idx, block_tsubs_indices, all_blocks, sigs, num_tsubs, num_uint64s = _precompute_block_sigs(v, k, t)
    
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    max_restarts = 50 # Can do more if fast
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomized Greedy
        # The greedy above picks max gain globally. 
        # To add randomness, we can perturb gains or pick from top-k.
        # For now, let's use the global max gain greedy.
        # To vary it, we can shuffle the blocks or use a different seed for rng.
        rng_restart = random.Random(42 + restart)
        
        blocks = _greedy_cover_numpy(v, k, t, rng_restart, deadline, tsub_to_idx, block_tsubs_indices, all_blocks, sigs, num_tsubs, num_uint64s)
        
        if blocks is None or not blocks:
            continue
            
        # Remove redundant blocks
        blocks = _remove_redundant_fast(blocks, v, t, tsub_to_idx, block_tsubs_indices, deadline)
        
        # Verify
        # Simple check
        covered = set()
        for block in blocks:
            covered.update(itertools.combinations(block, t))
        all_tsubs = set(itertools.combinations(range(v), t))
        if not all_tsubs.issubset(covered):
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # If we find a very good solution, break early?
            if best_count <= 74:
                break
                
    if best_blocks is None:
        # Fallback to simple greedy if numpy failed or timed out
        # This is a safety net
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            if time.time() > deadline:
                break
            start_tsub = min(uncovered)
            block = list(start_tsub)
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_point = -1
                block_t1_subs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = min(candidates) if candidates else 0
                block.append(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks

    return best_blocks
