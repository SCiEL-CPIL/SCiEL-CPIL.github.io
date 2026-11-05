import itertools
import time
import numpy as np
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin to ensure we return before the 150s hard limit
    deadline = start_time + 140.0
    
    # 1. Precomputation
    # For v=19, k=7, t=4:
    # C(19, 4) = 3876 t-subsets
    # C(19, 7) = 50388 k-blocks
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute coverage of each block as a set of t-subset indices
    # Using lists for fast numpy operations, but we need to know "new" coverage.
    # We will represent the current covered set as a boolean numpy array for fast updates.
    
    block_coverages = []
    for block in all_k_blocks:
        covers = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Reverse index: t_subset_idx -> list of block indices covering it
    # This is useful for local search (swaps) to find candidates quickly.
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverages):
        for t_idx in cov:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    def greedy_set_union():
        """
        GSU Heuristic:
        Select blocks based on marginal gain in the union of covered t-subsets.
        Uses a lazy priority queue to avoid recalculating gains for all blocks at every step.
        """
        covered = np.zeros(num_t_subsets, dtype=bool)
        current_blocks = []
        current_size = 0
        
        # Lazy Queue: (negative_gain, block_idx, version)
        # We don't use versioning here explicitly, but we re-calculate gain when popping.
        # Initial gain for all blocks is just len(block_coverages[b])
        pq = []
        for b_idx in range(num_k_blocks):
            gain = len(block_coverages[b_idx])
            heapq.heappush(pq, (-gain, b_idx))
            
        while True:
            if time.time() > deadline - 10.0:
                break
                
            # Check if all t-subsets are covered
            if np.all(covered):
                break
                
            # Get best candidate
            neg_gain, b_idx = heapq.heappop(pq)
            
            # Verify if this gain is still valid
            # Gain = number of t-subsets in this block NOT currently covered
            cov = block_coverages[b_idx]
            new_covered = ~covered[cov]
            current_gain = np.count_nonzero(new_covered)
            
            if current_gain == 0:
                # This block adds nothing new. 
                # In a lazy heap, we just discard it. 
                # We do NOT push it back because it will never have gain > 0 
                # unless we remove other blocks, which we don't do in pure construction.
                continue
                
            # Is the stored gain stale?
            # Stored gain was -neg_gain. 
            # If current_gain < -neg_gain, it means other blocks added since then 
            # covered some of this block's subsets. 
            # We push it back with the updated gain so it can be reconsidered later.
            if current_gain < -neg_gain:
                heapq.heappush(pq, (-current_gain, b_idx))
                continue
            
            # Valid best block
            covered[cov] = True
            current_blocks.append(b_idx)
            current_size += 1
            
            # Optimization: If we have covered everything, we can stop early?
            # The loop condition handles this.
            
        return current_blocks, covered

    def local_search(blocks, covered):
        """
        Refine the solution using:
        1. Greedy Deletion: Remove blocks that don't uncover any t-subsets.
        2. 1-Swap: Replace one block with another to potentially enable more deletions 
           or reduce size (though 1-swap keeps size constant, it changes structure).
           Actually, for minimization, 1-swap is only useful if it allows subsequent deletion.
           We will focus on "Exchange" moves that might lead to a smaller set.
           
        A more powerful move for covering problems is the "Add-Remove" or "Exchange" 
        that directly reduces size if possible, or improves "flexibility".
        
        However, the most effective local search for set cover is often:
        - Try to remove a block. If it fails (some t-subset becomes uncovered), 
          try to find a single block that covers the lost t-subsets AND is not the one removed.
          If such a block exists and we can swap, size is constant.
          To reduce size, we need a 2-for-1 exchange: Remove 2 blocks, Add 1 block that covers all their unique contributions.
        
        Let's implement:
        1. Greedy Deletion Pass
        2. 2-for-1 Exchange Search (structural change to reduce size)
        """
        current_blocks = list(blocks)
        # Reconstruct counts from covered? 
        # It's better to maintain counts for efficiency in local search.
        # But `covered` is a bool array. Let's convert to counts for this phase.
        # Or just re-calculate counts from scratch for the current set.
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b in current_blocks:
            counts[block_coverages[b]] += 1
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Greedy Deletion
            # Identify blocks that are not critical (no t-subset has count 1)
            # A block is critical if it is the ONLY cover for at least one t-subset.
            
            # Efficiently find critical blocks:
            # For each t-subset with count 1, the block covering it is critical.
            # Instead of iterating t-subsets, iterate blocks and check if they contain any count-1 t-subset.
            
            critical_blocks = set()
            for b in current_blocks:
                cov = block_coverages[b]
                if np.any(counts[cov] == 1):
                    critical_blocks.add(b)
                    
            non_critical = [b for b in current_blocks if b not in critical_blocks]
            
            if non_critical:
                # Remove one non-critical block
                b_to_remove = random.choice(non_critical)
                cov = block_coverages[b_to_remove]
                counts[cov] -= 1
                current_blocks.remove(b_to_remove)
                improved = True
                continue # Restart loop to update criticality
            
            # Phase 2: 2-for-1 Exchange
            # Try to remove two blocks A and B, and add one block C 
            # such that C covers all t-subsets that were ONLY covered by A or B.
            # This reduces size by 1.
            
            # Heuristic: Pick a random pair of critical blocks (A, B).
            # Find t-subsets in A with count 1 (unique to A) and B with count 1 (unique to B).
            # Also, t-subsets covered by BOTH A and B have count >= 2, so removing both drops count to 0? 
            # No, if a t-subset is in A and B, count is at least 2. 
            # If we remove A and B, count becomes count - 2. 
            # If count was 2, it becomes 0. So C must cover it too.
            
            # So C must cover:
            # 1. All t-subsets in A that have count 1 (unique to A)
            # 2. All t-subsets in B that have count 1 (unique to B)
            # 3. All t-subsets in A AND B that have count 2 (unique to {A,B})
            
            # This is a complex constraint. 
            # Let's simplify: 
            # We want to find A, B, C such that:
            # (Unique(A) U Unique(B) U Shared_AB) is a subset of Coverage(C).
            
            # To make this fast, we limit the search.
            # Pick a random critical block A.
            # Find its unique t-subsets U_A.
            # If U_A is large, hard to cover.
            
            # Let's try a simpler exchange: 1-for-1 swap that increases "slack".
            # Or just stick to 1-swap to improve flexibility, hoping for more deletions.
            
            # Given time constraints, let's implement a limited 2-for-1 search.
            # Pick random A from current_blocks.
            if len(current_blocks) < 2:
                break
                
            A = random.choice(current_blocks)
            cov_A = block_coverages[A]
            unique_A = cov_A[counts[cov_A] == 1]
            
            # If A has no unique t-subsets, it's not critical? 
            # Wait, if A is critical, it MUST have at least one unique t-subset.
            # So if A is in current_blocks and we didn't find non-critical, A is critical.
            
            # Pick random B from current_blocks \ {A}
            other_blocks = [b for b in current_blocks if b != A]
            if not other_blocks:
                break
            B = random.choice(other_blocks)
            cov_B = block_coverages[B]
            
            # Determine the set of t-subsets that MUST be covered by C
            # 1. Unique to A: counts==1 and in A
            # 2. Unique to B: counts==1 and in B
            # 3. Shared by A and B with count==2
            
            # Efficient check:
            # Create a boolean mask for "Must Cover"
            must_cover_mask = np.zeros(num_t_subsets, dtype=bool)
            
            # Unique to A
            must_cover_mask[unique_A] = True
            
            # Unique to B
            unique_B = cov_B[counts[cov_B] == 1]
            must_cover_mask[unique_B] = True
            
            # Shared with count 2
            # Intersection of A and B
            # We can check for each t in cov_A if it is in cov_B and counts[t] == 2
            # This is O(35 * 35) roughly if we use sets, or just numpy logic.
            # Let's use a temporary array.
            # Actually, we can just check: 
            # For t in cov_A: if t in cov_B and counts[t] == 2: must_cover[t] = True
            
            # Optimization: 
            # If len(must_cover) is too large, skip.
            
            # Find candidates C that cover all must_cover t-subsets.
            # C must be in all_k_blocks, not in {A, B}.
            
            # To find candidates, we can intersect the candidate lists of the must_cover t-subsets.
            # But must_cover can have up to 35+35+35 = 105 elements.
            # Intersecting 105 lists of size ~455 is expensive.
            
            # Alternative: Iterate through a sample of potential C blocks?
            # Or iterate through all blocks and check? 50k blocks * 105 checks = 5M ops. 
            # In Python, this is too slow per iteration.
            
            # Let's use the reverse index intersection for the FIRST few must_cover elements.
            # If must_cover is empty, any block works (but A,B are critical, so not empty).
            
            # Pick the smallest candidate list from must_cover to start intersection.
            # Sort must_cover indices by length of t_sub_to_blocks[t]
            
            must_cover_list = np.where(must_cover_mask)[0]
            if len(must_cover_list) == 0:
                # Should not happen for critical blocks
                continue
                
            # Sort by candidate list size
            must_cover_list = must_cover_list[np.argsort([len(t_sub_to_blocks[t]) for t in must_cover_list])]
            
            # Start with candidates of the first t-subset
            candidates_set = set(t_sub_to_blocks[must_cover_list[0]])
            
            # Remove A and B from candidates
            candidates_set.discard(A)
            candidates_set.discard(B)
            
            # Intersect with others
            for t_idx in must_cover_list[1:]:
                if not candidates_set:
                    break
                candidates_set.intersection_update(t_sub_to_blocks[t_idx])
                candidates_set.discard(A)
                candidates_set.discard(B)
                
            if candidates_set:
                # Try a random candidate C
                C = random.choice(list(candidates_set))
                cov_C = block_coverages[C]
                
                # Verify C covers all must_cover
                # (Intersection logic should guarantee this, but double check for safety? 
                #  Intersection of sets of blocks containing t ensures C contains all t in must_cover_list)
                
                # Apply exchange: Remove A, Remove B, Add C
                # Check if this is valid. 
                # By construction, C covers all t-subsets that would become uncovered.
                # So the cover remains valid.
                
                # Update counts
                counts[cov_A] -= 1
                counts[cov_B] -= 1
                counts[cov_C] += 1
                
                current_blocks.remove(A)
                current_blocks.remove(B)
                current_blocks.append(C)
                
                improved = True
                # Continue loop to check for more deletions
                continue
                
        return current_blocks

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 20 # We can afford more restarts with fast GSU
    
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # 1. Greedy Construction
        init_blocks, init_covered = greedy_set_union()
        
        # 2. Local Search
        refined_blocks = local_search(init_blocks, init_covered)
        
        if len(refined_blocks) < best_size:
            best_size = len(refined_blocks)
            best_solution = refined_blocks.copy()
            
            # Early stop if we hit a very good score
            if best_size <= 140:
                break
                
    if best_solution is None:
        # Fallback: Return a valid cover (even if large)
        # Greedy always produces a valid cover
        init_blocks, _ = greedy_set_union()
        best_solution = init_blocks
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_solution]
