import itertools
import random
import time
import numpy as np


def _remove_redundant_blocks(block_indices, block_tsubs_map, counts):
    """Remove blocks that do not uniquely cover any t-subset.
    block_indices: list of block indices
    block_tsubs_map: list of numpy arrays of tsub indices for each block
    counts: numpy array of counts for each tsub
    Returns: (new_block_indices, updated_counts)
    """
    if not block_indices:
        return [], counts

    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        to_remove = set()
        
        for b_idx in current_blocks:
            tsubs = block_tsubs_map[b_idx]
            # Check if all tsubs covered by this block have count > 1
            if np.all(counts[tsubs] > 1):
                to_remove.add(b_idx)
                changed = True
        
        if to_remove:
            for b_idx in to_remove:
                tsubs = block_tsubs_map[b_idx]
                counts[tsubs] -= 1
            
            current_blocks = [b for b in current_blocks if b not in to_remove]
            
    return current_blocks, counts


def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Map each block to the list of tsub indices it covers
    block_tsubs_map = []
    for b in all_kblocks:
        tsubs = []
        for ts in itertools.combinations(b, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs_map.append(np.array(tsubs, dtype=np.int32))
        
    # Map each tsub to the list of block indices that cover it
    tsub_blocks_map = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs_map):
        for ts_idx in tsubs:
            tsub_blocks_map[ts_idx].append(b_idx)

    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Greedy Construction with Randomization
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.monotonic() > deadline - 5.0:
            break
            
        counts = np.zeros(num_tsubs, dtype=np.int8)
        selected_blocks = []
        
        uncovered_mask = np.ones(num_tsubs, dtype=bool)
        num_uncovered = num_tsubs
        
        while num_uncovered > 0:
            if time.monotonic() > deadline - 2.0:
                break
            
            # Pick a random uncovered tsub to anchor the search
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break
            anchor_tsub = unc_idx[rng.randrange(len(unc_idx))]
            
            # Get candidates: blocks that cover anchor_tsub
            candidates = tsub_blocks_map[anchor_tsub]
            
            if not candidates:
                break
            
            # Evaluate candidates
            best_candidate = -1
            best_gain = -1
            
            for b_idx in candidates:
                tsubs = block_tsubs_map[b_idx]
                # Count how many of these tsubs are currently uncovered
                gain = np.sum(uncovered_mask[tsubs])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
            
            if best_candidate == -1:
                break
                
            # Select best_candidate
            selected_blocks.append(best_candidate)
            
            # Update counts and uncovered_mask
            tsubs = block_tsubs_map[best_candidate]
            counts[tsubs] += 1
            uncovered_mask[tsubs] = False
            num_uncovered -= best_gain
            
        # End of Greedy Phase
        
        current_blocks = selected_blocks
        
        # Refinement: Remove redundant blocks
        current_blocks, counts = _remove_redundant_blocks(current_blocks, block_tsubs_map, counts)
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]
            
    # Final Check: Ensure we have a valid solution
    if best_blocks is None:
        # Fallback: Deterministic greedy
        if time.monotonic() < deadline - 1.0:
            counts = np.zeros(num_tsubs, dtype=np.int8)
            selected_blocks = []
            uncovered_mask = np.ones(num_tsubs, dtype=bool)
            num_uncovered = num_tsubs
            
            while num_uncovered > 0:
                if time.monotonic() > deadline - 0.5:
                    break
                unc_idx = np.where(uncovered_mask)[0]
                if len(unc_idx) == 0:
                    break
                anchor_tsub = unc_idx[0]
                candidates = tsub_blocks_map[anchor_tsub]
                if not candidates:
                    break
                
                best_candidate = -1
                best_gain = -1
                for b_idx in candidates:
                    tsubs = block_tsubs_map[b_idx]
                    gain = np.sum(uncovered_mask[tsubs])
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_idx
                        
                if best_candidate == -1:
                    break
                    
                selected_blocks.append(best_candidate)
                tsubs = block_tsubs_map[best_candidate]
                counts[tsubs] += 1
                uncovered_mask[tsubs] = False
                num_uncovered -= best_gain
                
            current_blocks, counts = _remove_redundant_blocks(selected_blocks, block_tsubs_map, counts)
            best_blocks = current_blocks
            best_len = len(best_blocks)
    
    # Convert indices back to lists of points
    result_blocks = [list(all_kblocks[i]) for i in best_blocks]
    
    return result_blocks
