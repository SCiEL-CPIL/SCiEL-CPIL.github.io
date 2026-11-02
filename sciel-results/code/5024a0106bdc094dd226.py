import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Pre-compute all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    # Map each block to the list of t-subset indices it covers
    block_to_t_indices = {}
    for b in all_k_blocks:
        t_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_t_indices[b] = t_indices
        
    # Pre-compute mapping from t-subset index to list of block indices containing it
    # This is crucial for fast "can we cover L_ij?" checks
    t_to_blocks = defaultdict(list)
    for i, b in enumerate(all_k_blocks):
        for idx in block_to_t_indices[b]:
            t_to_blocks[idx].append(i)
            
    # 2. Greedy Construction with Weights
    # Weight of a t-subset: 1 / frequency in all blocks
    # This prioritizes "rare" subsets which are harder to cover
    freq = [0] * num_t
    for b in all_k_blocks:
        for idx in block_to_t_indices[b]:
            freq[idx] += 1
            
    weights = [1.0 / freq[i] if freq[i] > 0 else 0.0 for i in range(num_t)]
    
    uncovered = set(range(num_t))
    blocks_in_solution = [] # List of block tuples
    block_indices_in_solution = [] # List of indices into all_k_blocks
    
    # Greedy Loop
    while uncovered:
        if time.time() > deadline - 30.0:
            break
            
        best_block_idx = -1
        best_score = -1
        
        # To speed up, we don't necessarily need to check ALL blocks if we are smart,
        # but with 75k blocks, scanning them is O(75k * 56) ~ 4M ops per block selection.
        # 350 blocks * 4M = 1.4B ops. Too slow in Python.
        
        # Optimization: Only consider blocks that cover at least one uncovered subset.
        # We can maintain a list of candidate blocks.
        # Alternatively, just scan. Let's try to optimize the inner loop.
        
        # A faster greedy: Pick a random uncovered subset, then pick a block covering it that maximizes coverage.
        # Or: Pick the uncovered subset with the highest weight (hardest to cover), then pick the best block covering it.
        
        if uncovered:
            # Find the "hardest" uncovered subset (highest weight)
            # To avoid O(N) scan for max weight every time, we can just pick a random one 
            # or keep a priority queue. Given the small number of iterations, a simple scan 
            # of uncovered to find max weight is O(|uncovered|). |uncovered| starts at 11k.
            # 350 * 11k = 3.8M ops. This is fine.
            
            # However, finding the best block covering that specific subset is faster:
            # t_to_blocks[hardest_idx] gives candidates.
            
            # Let's try: Pick random uncovered subset, then best block covering it.
            # This is a standard heuristic (Randomized Greedy / RAS).
            
            # To be more robust, let's use the weighted score of the block.
            # We will scan a subset of blocks? No, let's stick to the "hardest subset" approach.
            # It's efficient because t_to_blocks[subset] is small (avg 11628*56/75582 ~ 8.6 blocks per subset).
            
            # Pick a random uncovered subset to break ties and add diversity
            # Or pick the one with max weight.
            # Let's pick the one with max weight among uncovered.
            max_w = -1
            target_idx = -1
            for idx in uncovered:
                if weights[idx] > max_w:
                    max_w = weights[idx]
                    target_idx = idx
            
            candidates = t_to_blocks[target_idx]
            
            best_score = -1
            best_candidate = -1
            for c_idx in candidates:
                # Score: sum of weights of uncovered subsets in this block
                score = 0.0
                for idx in block_to_t_indices[all_k_blocks[c_idx]]:
                    if idx in uncovered:
                        score += weights[idx]
                if score > best_score:
                    best_score = score
                    best_candidate = c_idx
            
            if best_candidate != -1:
                b = all_k_blocks[best_candidate]
                blocks_in_solution.append(b)
                block_indices_in_solution.append(best_candidate)
                # Update uncovered
                for idx in block_to_t_indices[b]:
                    if idx in uncovered:
                        uncovered.remove(idx)
                        
    # Ensure we have a valid covering if we broke early due to time?
    # If uncovered is not empty, we must finish.
    # If we are close to deadline, we might have an invalid covering.
    # The task requires a valid covering.
    # So we must finish the greedy phase.
    while uncovered:
        if time.time() > deadline - 10.0:
             # Emergency: just add blocks for remaining uncovered
             # This might be suboptimal but valid
             pass
             
        # Pick random uncovered
        target_idx = random.choice(list(uncovered))
        candidates = t_to_blocks[target_idx]
        
        best_score = -1
        best_candidate = -1
        for c_idx in candidates:
            score = 0.0
            for idx in block_to_t_indices[all_k_blocks[c_idx]]:
                if idx in uncovered:
                    score += weights[idx]
            if score > best_score:
                best_score = score
                best_candidate = c_idx
        
        b = all_k_blocks[best_candidate]
        blocks_in_solution.append(b)
        block_indices_in_solution.append(best_candidate)
        for idx in block_to_t_indices[b]:
            if idx in uncovered:
                uncovered.remove(idx)
                
    # 3. Local Search: Reduction
    # Maintain cover_count for each t-subset
    cover_count = [0] * num_t
    for b in blocks_in_solution:
        for idx in block_to_t_indices[b]:
            cover_count[idx] += 1
            
    def is_valid_removal(block_idx, cover_count_local):
        # Check if any subset in this block has count 1 (and that count is from this block)
        # Actually, we just check if any subset in this block has total count 1 in the current solution.
        # If we remove it, count becomes 0.
        for idx in block_to_t_indices[blocks_in_solution[block_idx]]:
            if cover_count_local[idx] == 1:
                return False
        return True

    # Pass 1: Remove unnecessary blocks
    changed = True
    while changed:
        if time.time() > deadline - 20.0:
            break
        changed = False
        # Iterate in reverse to avoid index shifting issues, or use a list of indices to remove
        to_remove = []
        for i in range(len(blocks_in_solution)):
            if is_valid_removal(i, cover_count):
                to_remove.append(i)
        
        if to_remove:
            # Remove them
            for i in sorted(to_remove, reverse=True):
                b = blocks_in_solution[i]
                for idx in block_to_t_indices[b]:
                    cover_count[idx] -= 1
                blocks_in_solution.pop(i)
                block_indices_in_solution.pop(i)
            changed = True

    # Pass 2: 2-for-1 Exchange
    # Try to replace two blocks with one
    # We iterate through pairs.
    # To speed up, we only consider pairs if their union of "unique" coverage is small.
    
    def get_unique_coverage(block_idx):
        """Returns set of t-subset indices covered ONLY by this block in current solution."""
        uniques = set()
        for idx in block_to_t_indices[blocks_in_solution[block_idx]]:
            if cover_count[idx] == 1:
                uniques.add(idx)
        return uniques

    # We will try to find pairs (i, j) such that Unique(i) U Unique(j) is covered by some block B_new.
    # Note: Unique(i) is relative to the *current* set.
    # If we remove i and j, the subsets in Unique(i) U Unique(j) become uncovered (count 0).
    # We need a block B_new that covers all of them.
    
    # Optimization: Pre-compute Unique(i) for all i?
    # This changes as we modify the set.
    # Let's do it on the fly but cache it?
    # Given N ~ 350, computing Unique(i) is O(56) per block. Total O(N*56) ~ 20k ops.
    # Doing this for every pair is O(N^2 * 56) ~ 60k * 56 = 3.3M ops per full pass.
    # This is acceptable if we don't do too many passes.
    
    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        n = len(blocks_in_solution)
        # Compute unique sets for all blocks
        unique_sets = [get_unique_coverage(i) for i in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Potential loss if we remove i and j
                lost = unique_sets[i] | unique_sets[j]
                
                # If lost is empty, we can just remove both (handled in Pass 1, but maybe Pass 1 missed it? 
                # No, Pass 1 removes single blocks. If both have empty unique sets, Pass 1 would have removed them one by one.
                # So lost is likely non-empty).
                
                if not lost:
                    # Remove both
                    for idx in lost: # empty
                        pass
                    # Actually, if lost is empty, it means no subset is uniquely covered by i or j.
                    # So we can remove them.
                    # Update counts
                    for idx in block_to_t_indices[blocks_in_solution[i]]:
                        cover_count[idx] -= 1
                    for idx in block_to_t_indices[blocks_in_solution[j]]:
                        cover_count[idx] -= 1
                    # Remove from list (reverse order)
                    blocks_in_solution.pop(j)
                    blocks_in_solution.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                
                # Check if there is a block covering all of 'lost'
                # 'lost' is a set of indices.
                # Find intersection of t_to_blocks[s] for s in lost.
                if len(lost) > 0:
                    # Start with blocks covering the first subset in lost
                    first_lost = next(iter(lost))
                    candidate_blocks = set(t_to_blocks[first_lost])
                    
                    for s in lost:
                        if not candidate_blocks:
                            break
                        candidate_blocks.intersection_update(t_to_blocks[s])
                    
                    if candidate_blocks:
                        # Found a block that covers all lost subsets
                        # Pick one. Ideally one that also covers many other things?
                        # For now, just pick the first one.
                        new_block_idx = next(iter(candidate_blocks))
                        
                        # Check if new_block_idx is already in the solution?
                        # If it is, we are just removing i and j.
                        # If it is not, we are adding it.
                        
                        # Apply change:
                        # 1. Add new block (if not present)
                        # 2. Remove i and j
                        
                        # Wait, if new_block_idx is already in the solution, we just remove i and j.
                        # If not, we add it.
                        
                        # Let's just add it and remove i, j.
                        # But if it's already there, we just remove i, j.
                        
                        # To keep it simple:
                        # Remove i and j from solution.
                        # Add new_block to solution.
                        
                        # Update counts for removal
                        for idx in block_to_t_indices[blocks_in_solution[i]]:
                            cover_count[idx] -= 1
                        for idx in block_to_t_indices[blocks_in_solution[j]]:
                            cover_count[idx] -= 1
                        
                        # Remove i and j
                        blocks_in_solution.pop(j)
                        blocks_in_solution.pop(i)
                        
                        # Add new block
                        blocks_in_solution.append(all_k_blocks[new_block_idx])
                        block_indices_in_solution.append(new_block_idx)
                        
                        # Update counts for addition
                        for idx in block_to_t_indices[all_k_blocks[new_block_idx]]:
                            cover_count[idx] += 1
                            
                        improved = True
                        break # Restart loop
            if improved:
                break
            if time.time() > deadline - 10.0:
                break

    return blocks_in_solution
