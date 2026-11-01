"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    
    blocks = list(itertools.combinations(range(v), k))
    
    # Map block tuple to list of t-subset indices it covers
    block_tsubs = {}
    for b in blocks:
        ts_idxs = []
        for ts in itertools.combinations(b, t):
            ts_idxs.append(tsub_idx[ts])
        block_tsubs[b] = np.array(ts_idxs, dtype=np.int32)
        
    # Map t-subset tuple to list of blocks containing it
    tsub_blocks = {ts: [] for ts in tsubs}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tsub_blocks[ts].append(b)
            
    return tsubs, tsub_idx, blocks, block_tsubs, tsub_blocks


def _greedy_cover(v, k, t, rng, precomp):
    """Randomized greedy set cover using precomputed data."""
    tsubs, tsub_idx, blocks, block_tsubs, tsub_blocks = precomp
    n_tsubs = len(tsubs)
    
    cov = np.zeros(n_tsubs, dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)
    
    selected_blocks = []
    
    while uncovered.any():
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered)[0]
        target_idx = uncovered_indices[rng.randrange(len(uncovered_indices))]
        target_tsub = tsubs[target_idx]
        
        # Get candidate blocks covering this t-subset
        candidates = tsub_blocks[target_tsub]
        
        best_block = None
        best_gain = -1
        
        for b in candidates:
            idxs = block_tsubs[b]
            # Calculate gain: number of currently uncovered t-subsets covered by b
            gain = np.sum(uncovered[idxs])
            if gain > best_gain:
                best_gain = gain
                best_block = b
        
        if best_block is None:
            # Should not happen if covering is possible
            break
            
        selected_blocks.append(best_block)
        
        # Update coverage
        idxs = block_tsubs[best_block]
        cov[idxs] += 1
        uncovered[idxs] = (cov[idxs] == 0)
        
    return selected_blocks


def _local_search(v, k, t, blocks, rng, precomp, time_limit=110.0):
    """Local search: delete redundant blocks, swap to escape local minima."""
    start_time = time.time()
    tsubs, tsub_idx, all_blocks, block_tsubs, tsub_blocks = precomp
    n_tsubs = len(tsubs)
    
    # Convert blocks to tuples for hashing
    blocks = [tuple(b) for b in blocks]
    
    # Build initial coverage
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for b in blocks:
        idxs = block_tsubs[b]
        cov[idxs] += 1
        
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to delete redundant blocks
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            b = blocks[i]
            idxs = block_tsubs[b]
            # If all t-subsets covered by b are covered by >1 block, b is redundant
            if np.all(cov[idxs] > 1):
                cov[idxs] -= 1
                blocks.pop(i)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to swap blocks to change structure
        # Pick a random block
        if not blocks:
            break
            
        i = rng.randrange(len(blocks))
        b = blocks[i]
        idxs_b = block_tsubs[b]
        
        # Find critical t-subsets (covered only by b)
        critical_mask = cov[idxs_b] == 1
        critical_idxs = idxs_b[critical_mask]
        
        if len(critical_idxs) == 0:
            # Should have been deleted in step 1, but maybe due to timing?
            # Or if we just swapped?
            # If no critical, it's redundant. Delete it.
            cov[idxs_b] -= 1
            blocks.pop(i)
            improved = True
            continue
        
        # Find a replacement block b' that covers all critical t-subsets
        # Candidates must cover the first critical t-subset
        first_crit_tsub = tsubs[critical_idxs[0]]
        candidates = tsub_blocks[first_crit_tsub]
        
        replacement = None
        for b_prime in candidates:
            if b_prime == b:
                continue
            idxs_bp = block_tsubs[b_prime]
            # Check if b' covers all critical t-subsets
            if np.all(np.isin(critical_idxs, idxs_bp)):
                replacement = b_prime
                break
        
        if replacement:
            # Perform swap
            idxs_bp = block_tsubs[replacement]
            cov[idxs_b] -= 1
            cov[idxs_bp] += 1
            blocks[i] = replacement
            improved = True
            continue
        
        # If no replacement found for this block, try another?
        # To avoid getting stuck, we can try a few random blocks.
        # But for simplicity, if we fail to improve, we break.
        # A more robust search would try multiple blocks.
        # Let's try a few more random blocks in this iteration.
        found_swap = False
        for _ in range(10): # Try 10 random blocks
            if time.time() - start_time > time_limit:
                break
            i = rng.randrange(len(blocks))
            b = blocks[i]
            idxs_b = block_tsubs[b]
            critical_mask = cov[idxs_b] == 1
            critical_idxs = idxs_b[critical_mask]
            
            if len(critical_idxs) == 0:
                cov[idxs_b] -= 1
                blocks.pop(i)
                improved = True
                found_swap = True
                break
            
            first_crit_tsub = tsubs[critical_idxs[0]]
            candidates = tsub_blocks[first_crit_tsub]
            
            for b_prime in candidates:
                if b_prime == b:
                    continue
                idxs_bp = block_tsubs[b_prime]
                if np.all(np.isin(critical_idxs, idxs_bp)):
                    cov[idxs_b] -= 1
                    cov[idxs_bp] += 1
                    blocks[i] = b_prime
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
                
        if not found_swap:
            # No improvement found
            break

    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Precompute
    precomp = _precompute(v, k, t)
    
    # Multiple randomized greedy restarts
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 10 # Increase restarts since greedy is faster
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rng = random.Random(r * 7919 + 13)
        blocks = _greedy_cover(v, k, t, rng, precomp)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on the best solution
    if best_blocks is not None:
        if time.time() < deadline - 5:
            rng = random.Random(42)
            remaining = min(110.0, deadline - time.time() - 3.0)
            best_blocks = _local_search(v, k, t, best_blocks, rng, precomp, time_limit=remaining)
    
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, precomp)
    
    # Ensure blocks are distinct and valid
    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
# EVOLVE-BLOCK-END
