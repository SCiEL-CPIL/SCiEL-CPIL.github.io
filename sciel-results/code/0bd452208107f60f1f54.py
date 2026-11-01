import itertools
import random
import time
import sys

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees for heuristic
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # Helper to get neighbors of a block (conflicting blocks)
    def get_neighbors(b_idx):
        neigh = set()
        for t_idx in block_t_indices[b_idx]:
            neigh.update(t_subset_blocks[t_idx])
        neigh.discard(b_idx)
        return neigh

    # 2. Branch and Bound for Maximum Independent Set (Packing)
    # We want to find a large set of blocks that are pairwise non-conflicting.
    
    best_packing = set()
    start_time = time.time()
    time_limit_bnb = 20.0 # Seconds
    
    # Global variables for B&B
    global_candidates = set(range(num_k_subsets))
    
    def branch_and_bound(candidates, current_set, depth=0):
        nonlocal best_packing
        nonlocal start_time
        nonlocal time_limit_bnb
        
        if time.time() - start_time > time_limit_bnb:
            return
            
        if not candidates:
            if len(current_set) > len(best_packing):
                best_packing = set(current_set)
            return
            
        # Pruning
        if len(current_set) + len(candidates) <= len(best_packing):
            return
            
        # Select a vertex to branch on
        # Heuristic: Pick a vertex with high degree in the candidate set
        # Approximate by picking one with high global degree or random
        # To avoid O(N^2) cost of calculating degree in candidates, 
        # we pick a random vertex or one with high global degree.
        # Let's pick a random vertex to ensure diversity, 
        # but bias towards high degree if possible.
        
        # For speed, let's just pick a random vertex from candidates.
        # Or, pick the one with max len(t_subset_blocks) sum?
        # Let's try picking a vertex that covers many "rare" t-subsets?
        # No, for packing, we want to remove as many candidates as possible.
        # So we want a vertex with many neighbors in `candidates`.
        
        # Sampling to find a high-degree vertex in candidates:
        # If candidates is large, sampling is good.
        sample_size = min(10, len(candidates))
        candidates_list = list(candidates)
        random.shuffle(candidates_list)
        sample = candidates_list[:sample_size]
        
        best_v = -1
        max_neigh_count = -1
        
        for v in sample:
            # Estimate degree in candidates
            # We can't easily count without iterating all candidates.
            # Let's just use the size of get_neighbors(v) as a proxy for global degree.
            # This is O(35 * 455) per sample. 10 samples -> 150k ops. Acceptable.
            neigh_v = get_neighbors(v)
            # Count how many are in candidates
            count = 0
            for n in neigh_v:
                if n in candidates:
                    count += 1
            if count > max_neigh_count:
                max_neigh_count = count
                best_v = v
                
        if best_v == -1:
            best_v = random.choice(candidates_list)
            
        # Branch 1: Include best_v
        # New candidates = candidates - neighbors(best_v) - {best_v}
        neigh_best_v = get_neighbors(best_v)
        new_candidates_include = set(candidates)
        new_candidates_include.difference_update(neigh_best_v)
        new_candidates_include.discard(best_v)
        
        # Update current set
        new_current_set = set(current_set)
        new_current_set.add(best_v)
        
        branch_and_bound(new_candidates_include, new_current_set, depth + 1)
        
        # Check time again
        if time.time() - start_time > time_limit_bnb:
            return
            
        # Branch 2: Exclude best_v
        new_candidates_exclude = set(candidates)
        new_candidates_exclude.discard(best_v)
        
        branch_and_bound(new_candidates_exclude, current_set, depth + 1)

    branch_and_bound(global_candidates, set())
    
    # If B&B didn't find anything (unlikely), fallback to empty
    if not best_packing:
        # Fallback to greedy packing if B&B failed or found nothing
        pool = set(range(num_k_subsets))
        while pool:
            # Pick random
            b = random.choice(list(pool))
            best_packing.add(b)
            neigh = get_neighbors(b)
            pool.difference_update(neigh)
            pool.discard(b)

    # 3. Phase 2: Solve Remaining Set Cover
    active_blocks = set(best_packing)
    cover_count = [0] * num_t_subsets
    
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        residual_uncovered = set(still_uncovered)
        
        # Greedy Set Cover
        # To speed up, we can precompute for each block the list of t-subsets it covers.
        # And check intersection with residual_uncovered.
        
        # We can iterate through all blocks and pick the one with max gain.
        # This is O(N * 35) per step.
        # Number of steps is at most num_t_subsets / min_gain.
        
        while residual_uncovered:
            best_block = -1
            max_gain = 0
            
            # Optimization: Only consider blocks that have at least one uncovered t-subset.
            # We can maintain a list of "active" blocks? 
            # Or just iterate all. 50k blocks * 35 checks = 1.75M ops.
            # If we need 1000 steps, that's 1.75B ops. Too slow in Python.
            
            # Better: Iterate through uncovered t-subsets, and for each, look at its blocks.
            # Pick the block that appears most often among the blocks covering the uncovered t-subsets.
            # This is the standard "greedy set cover" optimization.
            
            # Count frequency of each block in the candidate set
            # Candidates are blocks covering any t in residual_uncovered.
            
            # To avoid O(N) scan, we can just pick one uncovered t-subset, 
            # get its blocks, and pick the one with max "potential" (degree in residual).
            
            # Let's try: Pick a random uncovered t-subset.
            # Get all blocks covering it.
            # For each such block, count how many *other* uncovered t-subsets it covers.
            # Pick the best.
            
            if not residual_uncovered:
                break
                
            # Pick a random uncovered t-subset to anchor the search
            anchor_t = random.choice(list(residual_uncovered))
            candidate_blocks = t_subset_blocks[anchor_t]
            
            best_block = -1
            max_gain = 0
            
            for b_idx in candidate_blocks:
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                # Should not happen if cover is possible
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # 4. Minimize
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # 5. Local Search (Enhanced)
    start_time_ls = time.time()
    time_limit_ls = 90.0 # Leave 10s for overhead
    
    def try_remove(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx not in active_blocks:
            return False
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        active_blocks.remove(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] -= 1
        return True

    def try_swap(old_idx, new_idx):
        nonlocal active_blocks, cover_count
        if old_idx not in active_blocks or new_idx in active_blocks:
            return False
        
        unique_subsets = []
        for t_idx in block_t_indices[old_idx]:
            if cover_count[t_idx] == 1:
                unique_subsets.append(t_idx)
                
        if unique_subsets:
            for t_idx in unique_subsets:
                if t_idx not in block_t_indices[new_idx]:
                    return False

        active_blocks.remove(old_idx)
        active_blocks.add(new_idx)
        
        for t_idx in block_t_indices[old_idx]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    while time.time() - start_time_ls < time_limit_ls:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            if try_remove(B):
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                continue
            
            unique_subsets = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    unique_subsets.append(t_idx)
            
            if unique_subsets:
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                    if valid_candidates:
                        B_new = random.choice(valid_candidates)
                        if try_swap(B, B_new):
                            active_blocks, cover_count = minimize(active_blocks, cover_count)
                            improved = True
            else:
                pass
        
        if not improved:
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
