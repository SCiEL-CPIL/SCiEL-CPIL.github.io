import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each k-subset, store the list of t-subset indices it covers
    # And the set of points
    k_subsets = list(itertools.combinations(points, k))
    k_subset_to_t_indices = {}
    for ks in k_subsets:
        ts_list = [t_subset_to_idx[ts] for ts in itertools.combinations(ks, t)]
        k_subset_to_t_indices[ks] = ts_list

    def greedy_construct():
        uncovered = set(range(len(all_t_subsets)))
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to anchor the block
            anchor_idx = random.choice(list(uncovered))
            anchor_ts = all_t_subsets[anchor_idx]
            
            # Find all k-subsets that contain this t-subset
            # Optimization: Pre-compute for each t-subset, the list of k-subsets containing it?
            # That's 2024 * (21 choose 4) ~ 2024 * 5985 ~ 12M entries. Too much memory/time.
            # Instead, filter k_subsets on the fly? 346k checks per step. 76 steps. 26M checks.
            # Let's try to be smarter.
            
            # Actually, we can pre-compute `t_subsets_in_k`? No.
            # Let's just iterate through candidates.
            # To speed up, we can pre-compute for each point, which k-subsets contain it?
            # No, let's stick to the simple method first and see if it's fast enough.
            
            best_block = None
            best_gain = -1
            
            # Filter candidates: must contain anchor_ts
            # We can generate candidates by choosing k-t points from the remaining v-t points
            remaining_points = [p for p in points if p not in anchor_ts]
            for extra_points in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(anchor_ts + tuple(extra_points)))
                
                # Calculate gain
                gain = 0
                for t_idx in k_subset_to_t_indices[candidate_block]:
                    if t_idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break # Should not happen
                
            blocks.append(best_block)
            # Update uncovered
            for t_idx in k_subset_to_t_indices[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
            
            if time.time() > deadline:
                break
                
        return blocks

    def local_search(blocks):
        # Convert to set for faster lookup? No, we need to modify.
        # Use list.
        blocks = list(blocks)
        
        # Compute cover counts
        cover_counts = [0] * len(all_t_subsets)
        for b in blocks:
            for t_idx in k_subset_to_t_indices[b]:
                cover_counts[t_idx] += 1
                
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Find block with minimum unique coverage
            min_unique = float('inf')
            min_block_idx = -1
            min_block = None
            
            for i, b in enumerate(blocks):
                unique_count = 0
                for t_idx in k_subset_to_t_indices[b]:
                    if cover_counts[t_idx] == 1:
                        unique_count += 1
                if unique_count < min_unique:
                    min_unique = unique_count
                    min_block_idx = i
                    min_block = b
                    if min_unique == 0:
                        break
                        
            if min_unique == 0:
                # Remove the block
                b = blocks.pop(min_block_idx)
                for t_idx in k_subset_to_t_indices[b]:
                    cover_counts[t_idx] -= 1
                improved = True
            else:
                # Try to replace min_block with a better one
                # Find critical t-subsets
                critical_ts = [all_t_subsets[t_idx] for t_idx in k_subset_to_t_indices[min_block] if cover_counts[t_idx] == 1]
                
                # The new block must cover all points in critical_ts
                critical_points = set()
                for ts in critical_ts:
                    critical_points.update(ts)
                    
                if len(critical_points) > k:
                    # Cannot cover all critical points in one block
                    # Try a different strategy? Or just fail this step.
                    continue
                
                # Find candidate blocks that contain critical_points
                # And try to maximize coverage of other things?
                # Or just find ANY valid replacement.
                
                best_candidate = None
                # Heuristic: Try to find a block that covers critical_points and maximizes "stability"
                # For simplicity, just find one that covers critical_points.
                # To avoid infinite loops, we can prefer blocks that are "different" or cover more total t-subsets.
                
                # Generate candidates containing critical_points
                # Points to add: k - len(critical_points)
                remaining = [p for p in points if p not in critical_points]
                
                # If critical_points is empty, any block works? No, we want to replace min_block.
                # If critical_points is not empty, we must include it.
                
                # To limit search, if len(critical_points) is small, there are many candidates.
                # Let's just pick a random valid candidate? Or the one that covers most t-subsets?
                # Covering most t-subsets is good for robustness.
                
                best_gain = -1
                for extra in itertools.combinations(remaining, k - len(critical_points)):
                    candidate = tuple(sorted(critical_points | set(extra)))
                    # Check if candidate is valid (it is, by construction)
                    # Check if replacing min_block with candidate keeps validity
                    # This is expensive to check for all candidates.
                    # Let's just pick the first one? Or random?
                    # Random is safer to escape local minima.
                    best_candidate = candidate
                    break # Just take the first one? No, let's take a random one.
                
                if best_candidate is None:
                    continue
                    
                # Actually, let's try to find a candidate that is "good".
                # Good = covers many t-subsets that are currently covered >= 2 times?
                # Or just random.
                
                # Let's try a few random candidates
                found_replacement = False
                for _ in range(10): # Try 10 random candidates
                    extra = random.sample(remaining, k - len(critical_points))
                    candidate = tuple(sorted(critical_points | set(extra)))
                    
                    # Check validity of replacement
                    # Temporarily remove min_block, add candidate
                    # Check if any t_subset becomes 0
                    
                    # Optimization:
                    # t_subsets in min_block that are NOT in critical_ts are covered by others (count >= 2).
                    # t_subsets in critical_ts are covered by min_block (count == 1).
                    # So we just need to ensure candidate covers all critical_ts. (It does by construction).
                    # We also need to ensure that removing min_block doesn't drop any other t_subset to 0.
                    # This is true because we identified critical_ts as those with count 1.
                    # So any t_subset in min_block \ critical_ts has count >= 2.
                    # So removing min_block is safe for those.
                    # Adding candidate only increases counts.
                    # So the replacement is ALWAYS valid if candidate covers critical_points.
                    
                    # So we can just pick one!
                    best_candidate = candidate
                    found_replacement = True
                    break
                    
                if found_replacement:
                    # Perform replacement
                    blocks[min_block_idx] = best_candidate
                    # Update cover counts
                    for t_idx in k_subset_to_t_indices[min_block]:
                        cover_counts[t_idx] -= 1
                    for t_idx in k_subset_to_t_indices[best_candidate]:
                        cover_counts[t_idx] += 1
                    improved = True

        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(i) # Deterministic for reproducibility in test, but different each run
        blocks = greedy_construct()
        if not blocks:
            continue
            
        # Local search
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: return a valid covering even if not optimized
        # The greedy should always produce one.
        # If it didn't, we have a problem.
        return []
        
    # Convert tuples to lists for output
    return [list(b) for b in best_blocks]
