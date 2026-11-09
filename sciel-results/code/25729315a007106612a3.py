import itertools
import time
import random

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets and assign IDs
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    # Map each t-subset to an index
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # Precompute masks for each block
    # mask[b] is an integer with bits set for all t-subsets covered by block b
    masks = [0] * num_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask
        
    all_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = num_k_subsets + 1
    
    def greedy_construct(randomness_level):
        """
        Construct a covering using greedy heuristic.
        randomness_level: 0 for deterministic, >0 for randomized choice among top candidates.
        """
        uncovered = all_mask
        selected_indices = []
        
        while uncovered:
            # Find block with max gain
            max_gain = 0
            best_candidates = []
            
            # Optimization: If uncovered is small, we might finish soon.
            # Iterate all blocks
            for i in range(num_k_subsets):
                gain = (uncovered & masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if all_mask is fully covered by union of all blocks
            
            # Select candidate
            if randomness_level > 0 and len(best_candidates) > 1:
                # Pick randomly from top candidates
                # To make it more diverse, we can pick from top K, not just ties
                # But simple random from ties is a good start.
                # Let's try picking from top 10% of gains for more diversity
                # Re-calculate top 10%? Too slow.
                # Stick to random from ties for now, or introduce noise.
                chosen = random.choice(best_candidates)
            else:
                chosen = best_candidates[0]
                
            selected_indices.append(chosen)
            uncovered &= ~masks[chosen]
            
        return selected_indices

    def local_search(selected_indices):
        """
        Improve the covering by removing redundant blocks and swapping blocks.
        Returns the improved list of indices.
        """
        current_indices = list(selected_indices)
        
        # Helper to get mask of union of a set of indices
        def get_union_mask(indices):
            m = 0
            for i in indices:
                m |= masks[i]
            return m

        improved = True
        while improved:
            improved = False
            
            # 1. Redundancy Removal
            # Iterate in random order to avoid bias
            random.shuffle(current_indices)
            new_indices = []
            for i in current_indices:
                # Check if i is redundant
                # Union of all other blocks
                # Optimization: Calculate union of new_indices + remaining?
                # Simpler: Check if mask[i] is subset of union of all OTHER blocks.
                # We can maintain a union mask.
                pass # This logic is tricky to do in-place efficiently without recomputing union.
                
            # Efficient Redundancy Removal:
            # Maintain union_mask of current set.
            # To check if i is redundant:
            # Is (union_mask_without_i) & mask[i] == mask[i]?
            # union_mask_without_i = union_mask ^ (mask[i] & ~union_mask_without_i)? No.
            # We can just recompute union if the set is small (150).
            # Or, we can iterate and remove.
            
            # Let's do a pass:
            temp_indices = list(current_indices)
            changed = False
            # Sort by index for determinism in removal? No, random is better for escaping local optima.
            # But for redundancy, order doesn't matter much for correctness, just for which ones get removed.
            
            # Calculate union of all
            union_all = 0
            for i in temp_indices:
                union_all |= masks[i]
                
            # This doesn't help directly for individual removal.
            
            # Better approach for redundancy:
            # For each block i, calculate the mask covered by all OTHER blocks.
            # This is O(N^2) mask ops. N=150. 150*150 = 22500 ops. Very fast.
            
            to_remove = []
            n = len(temp_indices)
            for j in range(n):
                i = temp_indices[j]
                # Calculate union of all blocks except j
                union_others = 0
                for l in range(n):
                    if l != j:
                        union_others |= masks[temp_indices[l]]
                
                if (union_others & masks[i]) == masks[i]:
                    to_remove.append(j)
                    changed = True
                    # If we remove one, the union_others for others changes.
                    # So we must restart or be careful.
                    # Simplest: if any found, break and restart the while loop.
                    break
                    
            if changed:
                # Remove the first redundant one found
                current_indices.pop(to_remove[0])
                improved = True
                continue # Restart the while loop to recalculate redundancies
            
            # 2. 1-Exchange (Swap)
            # Try to replace a block i with a block i' that increases the "coverage margin"
            # Metric: popcount(mask[i'] & ~union_others_i) > popcount(mask[i] & ~union_others_i)
            # Note: union_others_i is the union of all blocks except i.
            
            swapped = False
            # Random order for i
            indices_to_try = list(range(len(current_indices)))
            random.shuffle(indices_to_try)
            
            for j in indices_to_try:
                i = current_indices[j]
                # Calculate union of others
                union_others = 0
                for l in range(len(current_indices)):
                    if l != j:
                        union_others |= masks[current_indices[l]]
                
                # Current contribution of i
                contrib_i = (masks[i] & ~union_others).bit_count()
                
                # Find best i'
                best_gain = contrib_i
                best_idx = -1
                # Scan all possible blocks
                # Optimization: Only consider blocks that cover at least one subset in ~union_others?
                # Actually, we want max (mask[x] & ~union_others).bit_count()
                
                # To speed up, we can pre-filter?
                # For now, scan all 50k.
                # This is the bottleneck. 150 * 50k = 7.5M bit_count ops.
                # Might be too slow if done in pure Python loop inside the while loop.
                # Let's limit the number of swaps attempted per local search pass.
                
                # Heuristic: Only try a few random candidates for i'?
                # Or, only try i' if it shares many points with i?
                # Let's try a limited scan: Top 1000 candidates by random or pre-sorted?
                # Pre-sorting blocks by "popularity" (total subsets covered) doesn't help for specific union_others.
                
                # Let's try a limited number of random swaps per pass.
                # Pick 10 random candidates for i'
                # Actually, scanning all 50k in Python is ~0.05s.
                # 150 blocks * 0.05s = 7.5s per local search pass.
                # If we do 10 local search passes, that's 75s.
                # Plus greedy time.
                # This might be too slow.
                
                # Optimization:
                # We can skip the full scan if we are tight on time.
                # Or, we can use a heuristic:
                # Find i' that maximizes overlap with i? No.
                
                # Let's try to optimize the scan.
                # We can't easily vectorize this in pure Python.
                # Let's reduce the frequency of swap attempts.
                # Only attempt swap if we have time.
                
                if time.time() > deadline - 5.0:
                    break
                
                # Try to find a better block
                # We'll sample 1000 random blocks to speed up
                sample_size = min(1000, num_k_subsets)
                candidates = random.sample(range(num_k_subsets), sample_size)
                
                for x in candidates:
                    if x in current_indices:
                        continue
                    gain_x = (masks[x] & ~union_others).bit_count()
                    if gain_x > best_gain:
                        best_gain = gain_x
                        best_idx = x
                        
                if best_idx != -1:
                    current_indices[j] = best_idx
                    swapped = True
                    improved = True
                    # Restart local search to check for new redundancies
                    break
                    
            if swapped:
                continue
                
        return current_indices

    # Main Loop
    # 1. Deterministic Greedy
    if time.time() < deadline:
        indices = greedy_construct(0)
        indices = local_search(indices)
        count = len(indices)
        if count < best_count:
            best_count = count
            best_blocks = [list(k_subsets[i]) for i in indices]
            
    # 2. Randomized Greedy Restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        # Increase randomness with restarts
        rand_level = min(restarts, 10)
        indices = greedy_construct(rand_level)
        
        # Quick local search (just redundancy removal first to save time?)
        # Let's do full local search but with time check inside.
        indices = local_search(indices)
        
        count = len(indices)
        if count < best_count:
            best_count = count
            best_blocks = [list(k_subsets[i]) for i in indices]
            
        # If we found 151, we can stop? No, we want to beat it.
        # If we find 150, stop?
        if best_count <= 150:
            break

    if best_blocks is None:
        # Fallback: return deterministic greedy result if no time for local search
        indices = greedy_construct(0)
        best_blocks = [list(k_subsets[i]) for i in indices]

    return best_blocks

# EVOLVE-BLOCK-END
