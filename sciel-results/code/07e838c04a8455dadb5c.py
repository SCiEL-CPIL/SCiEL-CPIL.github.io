import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(k_subsets)
    
    # Map block to list of t-subset indices it covers
    block_cover = []
    # Map t-subset index to list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(k_subsets):
        covered = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered.append(idx)
            subset_to_blocks[idx].append(b_idx)
        block_cover.append(covered)

    # 2. Greedy Initialization
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        best_block = -1
        max_gain = 0
        # Iterate through all blocks to find best gain
        # Optimization: We can skip blocks that are already in `blocks`? 
        # Greedy usually allows re-adding? No, blocks must be distinct.
        # So we should avoid adding a block already in `blocks`.
        current_blocks_set = set(blocks)
        
        for b_idx in range(num_k_subsets):
            if b_idx in current_blocks_set:
                continue
            # Calculate gain
            # Using set intersection
            gain = len(uncovered.intersection(block_cover[b_idx]))
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
                if gain == len(uncovered): # Perfect match
                    break
        
        if best_block == -1 or max_gain == 0:
            break # Should not happen if uncovered is not empty and we have blocks
            
        blocks.append(best_block)
        uncovered.difference_update(block_cover[best_block])

    # 3. Local Search
    # We will try to reduce the number of blocks.
    # State: `blocks` is a list of block indices.
    
    def get_unique_subsets(b_idx, blocks_list):
        """Return set of t-subset indices covered only by b_idx in blocks_list"""
        # Start with all subsets covered by b_idx
        unique = set(block_cover[b_idx])
        # Subtract subsets covered by other blocks
        for other_idx in blocks_list:
            if other_idx == b_idx:
                continue
            unique.difference_update(block_cover[other_idx])
            if not unique:
                break
        return unique

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant blocks
        i = 0
        while i < len(blocks):
            b_idx = blocks[i]
            unique = get_unique_subsets(b_idx, blocks)
            if not unique:
                blocks.pop(i)
                improved = True
                # Restart the pass? Or just continue? 
                # Continuing is fine, but indices shift. 
                # Let's just restart the whole local search loop to be safe and simple.
                break
            i += 1
        
        if improved:
            continue
            
        # Pass 2: Swap blocks
        # Try to find a block b_old that can be replaced by b_new 
        # such that b_new covers all unique subsets of b_old, 
        # and ideally makes the solution "better" (more redundant).
        
        # To speed up, we can shuffle the order of blocks we try to swap.
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            b_old = blocks[i]
            unique_old = get_unique_subsets(b_old, blocks)
            
            if not unique_old:
                # Should have been removed in Pass 1, but just in case
                blocks.pop(i)
                improved = True
                break
                
            # Find candidate b_new
            # b_new must cover all subsets in unique_old
            # We find the intersection of subset_to_blocks for each subset in unique_old
            
            # Sort unique_old by the number of blocks covering them (smallest first) to reduce intersection size early
            sorted_unique = sorted(unique_old, key=lambda x: len(subset_to_blocks[x]))
            
            candidates = None
            for s_idx in sorted_unique:
                blocks_covering_s = set(subset_to_blocks[s_idx])
                if candidates is None:
                    candidates = blocks_covering_s
                else:
                    candidates.intersection_update(blocks_covering_s)
                    if not candidates:
                        break
            
            if not candidates:
                continue
                
            # Filter out blocks already in the solution (except b_old itself, but we are replacing it)
            current_blocks_set = set(blocks)
            valid_candidates = [c for c in candidates if c not in current_blocks_set or c == b_old]
            
            if not valid_candidates:
                continue
                
            # Score candidates
            # We want a candidate that covers many subsets that are ALREADY covered by other blocks.
            # This increases redundancy.
            # Let's calculate the "redundancy gain".
            # Union of covers of all other blocks in the solution.
            # This is expensive to compute for every candidate.
            
            # Approximation: 
            # For each candidate, check if it covers the unique_old. (Already done)
            # Then, check how many of its subsets are covered by the union of OTHER blocks.
            # Let's just pick the candidate that maximizes the overlap with a random sample of other blocks?
            # Or simply: Pick the candidate that covers the most total subsets?
            # All candidates cover unique_old.
            # Let's pick the one that minimizes the number of NEW unique subsets it introduces?
            # Actually, we want to maximize the chance that other blocks become redundant.
            
            # Heuristic: Pick the candidate that has the maximum intersection with the union of covers of other blocks.
            # To make this fast, we can just pick the first valid candidate? 
            # That might be bad.
            
            # Let's try a few candidates.
            # Sort candidates by length of block_cover? No, all are size C(7,4)=35.
            
            # Let's just pick a random valid candidate to introduce diversity, 
            # or the one that appears most frequently in subset_to_blocks of unique_old?
            
            # Better Heuristic: 
            # Calculate for each candidate the number of subsets it covers that are NOT in unique_old 
            # and are covered by at least one other block.
            
            # This is still O(N_blocks * N_subsets).
            
            # Given time limits, let's just pick the first valid candidate from the sorted list 
            # (sorted by index or something deterministic) to ensure progress, 
            # or random to escape local minima.
            
            # Let's try to find a candidate that covers the most "shared" subsets.
            # We can precompute a "popularity" score for blocks?
            
            # For now, let's just swap with a random valid candidate.
            # This is a stochastic local search.
            b_new = random.choice(valid_candidates)
            
            # Perform swap
            blocks[i] = b_new
            improved = True
            break # Restart loop to check for new redundancies

    # 4. Final Cleanup: Remove any remaining redundant blocks
    # (The loop above might have exited due to time, leaving redundancies)
    for _ in range(len(blocks)):
        if time.time() > deadline:
            break
        i = 0
        removed = False
        while i < len(blocks):
            b_idx = blocks[i]
            unique = get_unique_subsets(b_idx, blocks)
            if not unique:
                blocks.pop(i)
                removed = True
            else:
                i += 1
        if not removed:
            break

    # Convert block indices back to lists of points
    result = [list(k_subsets[b]) for b in blocks]
    return result
