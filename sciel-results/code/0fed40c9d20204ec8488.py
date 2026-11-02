import itertools
import random
import time

def construct(v, k, t):
    # Precompute all k-subsets and their t-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_subset_to_t_subsets = {}
    for ks in all_k_subsets:
        ts = list(itertools.combinations(ks, t))
        k_subset_to_t_subsets[ks] = ts
    
    # Map t-subset to list of k-subsets containing it
    t_subset_to_k_subsets = {ts: [] for ts in itertools.combinations(range(v), t)}
    for ks in all_k_subsets:
        for ts in k_subset_to_t_subsets[ks]:
            t_subset_to_k_subsets[ts].append(ks)

    def greedy_construction(noise_scale=0.0):
        all_t_subsets = list(itertools.combinations(range(v), t))
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to break ties and add diversity
            seed_subset = random.choice(list(uncovered))
            
            # Generate all k-subsets containing seed_subset
            remaining_points = [p for p in range(v) if p not in seed_subset]
            need_points = k - t
            
            best_score = -1
            best_block = None
            
            # To speed up, we don't iterate ALL candidates if we can, 
            # but for v=19, k=7, t=4, candidates per seed are C(15, 3) = 455. 
            # This is small.
            for combo in itertools.combinations(remaining_points, need_points):
                block = tuple(sorted(seed_subset + combo))
                # Count how many uncovered subsets this block covers
                count = 0
                for ts in k_subset_to_t_subsets[block]:
                    if ts in uncovered:
                        count += 1
                
                if count == 0:
                    continue
                    
                # Add noise for randomization
                score = count + random.gauss(0, noise_scale)
                
                if score > best_score:
                    best_score = score
                    best_block = block
            
            if best_block is None:
                # Fallback: pick any block containing seed_subset
                # This shouldn't happen if uncovered is not empty
                break
                
            blocks.append(list(best_block))
            
            # Update uncovered
            for ts in k_subset_to_t_subsets[best_block]:
                uncovered.discard(ts)
                
        return blocks

    def local_search(blocks):
        # Convert to set of tuples for fast lookup
        current_blocks = set(tuple(b) for b in blocks)
        
        def get_coverage_counts(block_set):
            counts = {}
            for b in block_set:
                for ts in k_subset_to_t_subsets[b]:
                    counts[ts] = counts.get(ts, 0) + 1
            return counts

        def is_valid(block_set):
            # Check if all t-subsets are covered
            # We can just check if counts[ts] > 0 for all ts
            # But we need to be careful about memory/time.
            # Since we start from a valid state and only do safe moves, we might not need full check
            # unless we do risky moves. Here we only do deletions and swaps that preserve coverage.
            return True

        improved = True
        while improved:
            improved = False
            
            # 1. Deletion Phase
            counts = get_coverage_counts(current_blocks)
            to_remove = []
            for b in current_blocks:
                redundant = True
                for ts in k_subset_to_t_subsets[b]:
                    if counts[ts] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            
            if to_remove:
                for b in to_remove:
                    current_blocks.remove(b)
                    for ts in k_subset_to_t_subsets[b]:
                        counts[ts] -= 1
                improved = True
                # Restart loop as counts changed significantly
                continue 

            # 2. Swap Phase
            # For each block, see if we can replace it with another block that covers its unique subsets
            # and potentially allows further deletions.
            # To avoid infinite loops, we track visited states or limit swaps per iteration.
            # Simple strategy: Try to swap each block once.
            
            blocks_list = list(current_blocks)
            swapped = False
            
            for b in blocks_list:
                # Find unique subsets of b
                unique_ts = []
                for ts in k_subset_to_t_subsets[b]:
                    if counts[ts] == 1:
                        unique_ts.append(ts)
                
                if not unique_ts:
                    # Should have been removed in deletion phase
                    continue
                
                # Find candidate blocks that cover all unique_ts
                # Start with all blocks covering the first unique subset
                candidates = set(t_subset_to_k_subsets[unique_ts[0]])
                candidates.discard(b) # b is already in set, and we want a *different* block
                
                # Intersect with blocks covering other unique subsets
                for ts in unique_ts[1:]:
                    candidates &= set(t_subset_to_k_subsets[ts])
                    if not candidates:
                        break
                
                # Filter out blocks already in current_blocks
                candidates -= current_blocks
                
                if candidates:
                    # Pick a random candidate (or one with high coverage of other things?)
                    # For now, pick random to explore
                    new_block = random.choice(list(candidates))
                    
                    # Perform swap
                    current_blocks.remove(b)
                    current_blocks.add(new_block)
                    
                    # Update counts
                    for ts in k_subset_to_t_subsets[b]:
                        counts[ts] -= 1
                    for ts in k_subset_to_t_subsets[new_block]:
                        counts[ts] = counts.get(ts, 0) + 1
                    
                    swapped = True
                    # Restart deletion phase immediately as this swap might make others redundant
                    improved = True
                    break # Break inner loop, restart outer while loop
            
            if not swapped and not improved:
                # No more swaps or deletions
                pass
                
        return list(current_blocks)

    best_blocks = None
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Run multiple iterations with increasing noise or just random restarts
    # Greedy is deterministic if noise=0. We add noise to diversify.
    
    for i in range(50): # Limit iterations
        if time.time() - start_time > max_time:
            break
            
        # Increase noise slightly over time to explore more?
        # Or just use fixed noise.
        noise = 1.0 + i * 0.1
        
        greedy_sol = greedy_construction(noise_scale=noise)
        refined_sol = local_search(greedy_sol)
        
        if best_blocks is None or len(refined_sol) < len(best_blocks):
            best_blocks = refined_sol
            
    # If no iterations ran (unlikely), do a standard greedy
    if best_blocks is None:
        best_blocks = greedy_construction(noise_scale=0.0)
        best_blocks = local_search(best_blocks)
        
    return best_blocks
