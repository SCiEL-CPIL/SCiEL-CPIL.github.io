import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    # The evaluator kills after 150s. We leave a margin for safety and cleanup.
    deadline = start_time + 140.0

    # 1. Precomputation
    # Generate all t-subsets (triples)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Generate all k-subsets (blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute for each k-subset, the indices of the t-subsets it covers
    # This allows O(1) lookup of coverage for a block, and vectorized gain calculation
    # t_count is C(k, t)
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))
    
    # block_t_indices[i, j] = index of the j-th t-subset in the i-th k-subset
    block_t_indices = np.zeros((n_ksubs, t_count), dtype=np.int32)
    for i, block in enumerate(all_ksubs):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = tsub_index[sub]

    # Helper to convert block indices to actual blocks (lists of points)
    def indices_to_blocks(indices):
        return [list(all_ksubs[i]) for i in indices]

    def greedy_set_cover(seed):
        """
        Randomized Greedy Set Cover.
        Repeatedly picks the block that covers the most uncovered triples.
        To add diversity, it picks randomly from the top 'top_k' gainers.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_tsubs, dtype=bool)
        selected_indices = []
        
        # We can precompute the "inverse" mapping? No, that's huge.
        # But we can track which blocks are still "useful" or just filter candidates.
        # For v=24, k=7, n_ksubs is 346104. This is manageable.
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains for all blocks
            # covered[block_t_indices] gives a boolean matrix (n_ksubs, t_count)
            # ~ inverts to "uncovered"
            # sum(axis=1) counts newly covered triples per block
            # This is the bottleneck, but vectorized NumPy is fast.
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # No block covers any new triple. 
                # If not all covered, we failed to find a covering (should not happen for small v)
                # If all covered, we are done.
                if not np.all(covered):
                    return None
                else:
                    break
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # To escape local optima, we can sometimes pick from top-k instead of just max.
            # However, strict greedy is often good for coverings. 
            # Let's stick to picking randomly from the set of max-gain blocks.
            # If there are many, this adds diversity.
            
            best_idx = candidates[rng.randint(len(candidates))]
            selected_indices.append(best_idx)
            
            # Update covered status
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_indices

    def reduce_redundancy(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        We iterate until no more blocks can be removed.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        # We need to be careful with time here. 
        # The number of blocks is small (~70-80).
        # n_tsubs is 2024.
        
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage for each t-subset
            coverage_count = np.zeros(n_tsubs, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                # Check if this block is redundant
                subs_in_block = block_t_indices[b_idx]
                
                # If all these t-subsets have coverage_count > 1, the block is redundant
                # (Note: coverage_count includes this block itself)
                if np.all(coverage_count[subs_in_block] > 1):
                    # Redundant: do not add to new_blocks
                    # Decrement coverage for these subsets
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    # Keep it
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    def verify(blocks):
        """Check if all t-subsets are covered."""
        if not blocks:
            return False
        covered_bits = np.zeros(n_tsubs, dtype=bool)
        for block in blocks:
            idx = all_ksubs.index(tuple(block)) # This is slow, better to pass indices
            # Actually, verify is only called on the final best, so speed is less critical, 
            # but let's use indices if possible. 
            # The function signature returns blocks (lists). 
            # For verification, we can just re-calculate coverage.
            pass
        
        # Efficient verification using the precomputed indices if we had them, 
        # but here we just have lists.
        # Let's just do a simple check.
        covered_set = set()
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered_set.add(tsub)
        return len(covered_set) == n_tsubs

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    # The vectorized gain calculation is the key speedup.
    # Each step of greedy takes ~ O(n_ksubs * t_count) operations.
    # With ~80 blocks, that's ~80 * 346k * 35 ~ 1 billion ops? No, NumPy is C-speed.
    # It should be very fast.
    
    num_runs = 0
    # We will run as many as we can before deadline.
    while time.time() < deadline - 15.0:
        seed = num_runs
        block_indices = greedy_set_cover(seed)
        
        if block_indices is not None:
            # Reduce redundancy
            reduced_indices = reduce_redundancy(block_indices)
            
            if len(reduced_indices) < best_size:
                best_size = len(reduced_indices)
                best_blocks = reduced_indices
                
                # Early stop if we hit the known best (76) or lower
                # The task says best known is 76.
                if best_size <= 76:
                    # We can try to improve further, but 76 is a strong target.
                    # Let's run a few more to see if we can get 75? Unlikely but possible.
                    # However, 76 is the current best known, so matching it is a success.
                    # Breaking early might be safer to ensure we return a valid result.
                    # But maybe we can find 75?
                    # Let's not break immediately, allow a few more runs if time permits.
                    pass
        
        num_runs += 1
        
        # Safety break if we've done enough
        if num_runs > 1000:
            break

    if best_blocks is not None:
        return indices_to_blocks(best_blocks)
    
    # Fallback: if no valid covering found (should not happen), return empty or a trivial one?
    # The contract says "Valid when every t-subset...". Returning [] is invalid.
    # But greedy_set_cover should always find one.
    # If it fails due to timeout, we might have partial results? 
    # The function returns None on timeout.
    # Let's try one deterministic run if we have time, or just return empty.
    # Actually, if greedy_set_cover times out, it returns None.
    # If we never get a valid one, we are in trouble.
    # But for v=24, it should be fast.
    
    return []

