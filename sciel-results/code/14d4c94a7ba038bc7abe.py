import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    # Precompute t-subsets for each k-subset
    k_to_t_subsets = {
        block: set(itertools.combinations(block, t))
        for block in all_k_subsets
    }
    
    # All t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts depends on time
    # Let's do a few restarts with different random seeds for tie-breaking
    for restart in range(20):
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            # Iterate through all possible k-subsets
            # Optimization: We can prune candidates that don't intersect uncovered?
            # No, we need to find the max intersection.
            for block in all_k_subsets:
                # Calculate gain: number of t-subsets in block that are in uncovered
                # This is expensive if done naively for 50k blocks * 150 steps.
                # 50k * 150 = 7.5M checks. Each check is set intersection.
                # Set intersection of size ~35 (C(7,4)=35) with a large set.
                # This might be too slow in pure Python.
                
                # Optimization:
                # Instead of iterating all blocks, we can pick a random uncovered t-subset S.
                # The new block MUST contain S.
                # Number of blocks containing S is C(v-t, k-t) = C(15, 3) = 455.
                # This is much smaller!
                
                pass # We will implement the "pick uncovered S" strategy below.
            
            # Better Strategy:
            # 1. Pick a random uncovered t-subset S.
            # 2. Consider only k-subsets that contain S.
            # 3. Among those, pick the one with max gain.
            
            # Let's re-implement the loop body with this strategy.
            # We need to break out of the previous loop structure.
            
            # Actually, let's just use the "Pick Random Uncovered S" heuristic directly.
            
            # Pick a random uncovered t-subset
            # Using random.choice on a set is O(1) in Python 3.7+? No, sets are not sequences.
            # We can convert to list, but that's O(N).
            # Better: keep a list of uncovered, or use a heap?
            # For now, let's just pick the first one or use a random index if we maintain a list.
            # Maintaining a list of uncovered is better for random selection.
            
            # Let's restart the logic with a list for uncovered to allow O(1) random access.
            # But we need fast removal. A set is good for removal.
            # Compromise: Use a set for membership, and a list for iteration/random pick?
            # Or just pick the min uncovered? The seed does that.
            # Let's try picking a random one from a list, and sync the set.
            
            # This is getting complex. Let's stick to the "Pick Min Uncovered" but optimize the candidate search.
            # If we pick min uncovered S, we only check blocks containing S.
            
            target_subset = min(uncovered)
            
            # Find all k-subsets containing target_subset
            # This is C(v-t, k-t) = C(15, 3) = 455 blocks.
            # This is very fast.
            
            # Generate all k-subsets containing target_subset
            # target_subset has size t.
            # We need to choose k-t points from the remaining v-t points.
            remaining_points = [p for p in range(v) if p not in target_subset]
            
            best_gain = -1
            best_block = None
            
            for extra_points in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_subset + extra_points))
                
                # Calculate gain
                gain = 0
                for t_sub in k_to_t_subsets[block]:
                    if t_sub in uncovered:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(list(best_block))
            uncovered.difference_update(k_to_t_subsets[best_block])
            
        # End of greedy construction for this restart
        
        if not uncovered:
            # Reduction Phase
            # Check for redundant blocks
            # We need to count how many blocks cover each t-subset.
            # Or simply: for each block, check if removing it leaves any uncovered.
            
            # Efficient reduction:
            # 1. Count coverage of each t-subset by the current blocks.
            # 2. If a block has a t-subset with count 1, it's essential.
            # 3. If no t-subset in a block is essential, remove it.
            
            # Let's implement this.
            t_subset_counts = {}
            for block in blocks:
                for t_sub in k_to_t_subsets[tuple(block)]:
                    t_subset_counts[t_sub] = t_subset_counts.get(t_sub, 0) + 1
            
            # Now, a block is redundant if none of its t-subsets have count 1.
            new_blocks = []
            for block in blocks:
                essential = False
                for t_sub in k_to_t_subsets[tuple(block)]:
                    if t_subset_counts[t_sub] == 1:
                        essential = True
                        break
                if essential:
                    new_blocks.append(block)
            
            # We can repeat this reduction a few times
            for _ in range(5):
                if not new_blocks:
                    break
                t_subset_counts = {}
                for block in new_blocks:
                    for t_sub in k_to_t_subsets[tuple(block)]:
                        t_subset_counts[t_sub] = t_subset_counts.get(t_sub, 0) + 1
                
                filtered_blocks = []
                for block in new_blocks:
                    essential = False
                    for t_sub in k_to_t_subsets[tuple(block)]:
                        if t_subset_counts.get(t_sub, 0) == 1:
                            essential = True
                            break
                    if essential:
                        filtered_blocks.append(block)
                
                if len(filtered_blocks) == len(new_blocks):
                    break
                new_blocks = filtered_blocks
            
            if len(new_blocks) < best_count:
                best_count = len(new_blocks)
                best_blocks = new_blocks
                # If we reached a very good number, we can stop early
                if best_count <= 140: # Heuristic threshold
                    break
        else:
            # Greedy failed to cover all? Should not happen with this logic.
            pass

    if best_blocks is None:
        # Fallback to seed behavior if something went wrong
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    return best_blocks
