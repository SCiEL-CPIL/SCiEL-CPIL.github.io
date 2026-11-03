import itertools
import random
import time

def construct(v, k, t):
    # Parameters for the specific task: v=19, k=8, t=5
    # But the function signature is generic. We should handle the specific case efficiently 
    # or a generic case if parameters differ. 
    # The prompt implies this is for the specific task C(19,8,5).
    
    if v != 19 or k != 8 or t != 5:
        # Fallback to a simple greedy for other parameters if needed, 
        # though the task is specific.
        return _generic_greedy(v, k, t)

    # Precomputation for v=19, k=8, t=5
    points = range(v)
    
    # Generate all possible blocks (8-subsets)
    all_blocks = list(itertools.combinations(points, k))
    
    # Generate all possible t-subsets (5-subsets) and map them to an index
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each block, precompute the list of indices of t-subsets it covers
    block_t_indices = []
    for block in all_blocks:
        indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)

    num_t_subsets = len(all_t_subsets)
    
    def greedy_run(seed):
        random.seed(seed)
        # uncovered[i] = 1 if t-subset i is not yet covered
        uncovered = [1] * num_t_subsets
        covered_count = 0
        blocks_used = []
        
        # To speed up, we can maintain a count of how many uncovered subsets each block covers.
        # But recalculating gain for all 75k blocks every step is expensive.
        # Alternative: 
        # 1. Pick a random uncovered t-subset.
        # 2. Find all blocks that cover it.
        # 3. Among those, pick the one that covers the most *other* uncovered t-subsets.
        # This is the standard "Greedy Set Cover" heuristic: 
        # "Pick an uncovered element, then pick the set that covers it and maximizes other coverage."
        
        # Let's implement the "Pick uncovered element" strategy as it's faster than scanning all blocks.
        # We need a mapping: t_subset_idx -> list of block indices that cover it.
        
        # Precompute this mapping once outside? 
        # It depends on the specific run? No, the blocks are fixed.
        # But the "uncovered" status changes.
        
        # Mapping: t_idx -> list of block_indices
        t_to_blocks = [[] for _ in range(num_t_subsets)]
        for b_idx, indices in enumerate(block_t_indices):
            for t_idx in indices:
                t_to_blocks[t_idx].append(b_idx)
        
        # For efficiency, we can just use the set of uncovered indices.
        # At each step:
        # 1. Pick a random t_idx from uncovered.
        # 2. Get candidate blocks = t_to_blocks[t_idx].
        # 3. For each candidate block, count how many of its t-indices are in uncovered.
        # 4. Pick the block with max count.
        # 5. Add block to solution.
        # 6. Mark its t-indices as covered.
        
        # This is much faster.
        # Number of blocks covering a specific 5-subset:
        # We need to choose 3 more points from the remaining 14.
        # C(14, 3) = 364 blocks cover any specific 5-subset.
        # So we check 364 blocks per step.
        # Each check involves iterating over 56 t-subsets of the block and checking if they are uncovered.
        # 364 * 56 ~ 20,000 operations per step.
        # ~300 steps -> 6,000,000 operations. Very fast.
        
        # We can add randomization:
        # Instead of picking the absolute max, pick from top K or add noise.
        
        while covered_count < num_t_subsets:
            # Find an uncovered t-subset
            # Using a set of uncovered indices for O(1) pick? 
            # We can maintain a list and filter, or a set.
            # Since we remove elements, a set is good.
            # But we need to pick randomly from the set.
            
            # To avoid O(N) scan for an uncovered one, we can just iterate 
            # or use a random index and check if uncovered, retry if not.
            # Since coverage increases, the probability of hitting an uncovered one decreases.
            # Better: maintain a list of uncovered indices and pop randomly?
            # But we need to remove multiple indices.
            
            # Let's use a set `uncovered_set` for membership and a list `uncovered_list` for random access?
            # Or just pick a random index 0..num_t_subsets-1 and check if uncovered.
            # As coverage gets high, this becomes inefficient.
            
            # Alternative: 
            # Keep a list of uncovered indices.
            # At each step, pick a random index from this list.
            # When we cover new subsets, we don't remove them from the list immediately, 
            # but when we pick one, we check if it's still uncovered. If not, pick another.
            
            uncovered_indices = set(range(num_t_subsets))
            
            while uncovered_indices:
                # Pick a random uncovered subset
                # Converting set to list is O(N). 
                # We can maintain a list and clean it up periodically.
                # For simplicity and speed in Python, let's just use a list and filter.
                # Or better: 
                # Pick a random element from the set. 
                # Python 3.7+ doesn't have random.choice on set.
                # We can use next(iter(uncovered_indices)) but that's deterministic order (hash based).
                # To randomize:
                # items = list(uncovered_indices)
                # target = random.choice(items)
                # This is O(N) per step. N=11628. 300 steps. 3.5M ops. Fine.
                
                target_idx = random.choice(list(uncovered_indices))
                
                # Get candidate blocks
                candidates = t_to_blocks[target_idx]
                
                best_block = -1
                best_gain = -1
                best_covered = []
                
                for b_idx in candidates:
                    # Calculate gain: how many of this block's t-subsets are in uncovered_indices
                    gain = 0
                    covered_in_block = []
                    for t_idx in block_t_indices[b_idx]:
                        if t_idx in uncovered_indices:
                            gain += 1
                            covered_in_block.append(t_idx)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        best_covered = covered_in_block
                        
                # Add noise/randomization to break ties or explore
                # If we want to be smarter, we can consider blocks with gain == best_gain or best_gain - 1
                # But let's stick to greedy first, maybe with a small random restart logic.
                
                # Actually, pure greedy might get stuck in a local optimum regarding the *number* of blocks?
                # No, greedy set cover is an approximation algorithm. It doesn't "get stuck" in the sense of failing to cover, 
                # but it might use more blocks than necessary.
                # To improve, we can do multiple runs with different random seeds (for the tie-breaking or subset picking).
                
                # In this specific "pick target" greedy, the randomness comes from which uncovered subset we pick first.
                # Different choices lead to different block selections.
                
                if best_block == -1:
                    # Should not happen
                    break
                
                blocks_used.append(all_blocks[best_block])
                
                # Update uncovered
                for t_idx in best_covered:
                    uncovered_indices.remove(t_idx)
                
                # Early exit check for time? 
                # The caller handles the time limit by killing the process, 
                # but we can add a check inside to return best so far?
                # The prompt says "evaluator KILLS the program after 150s".
                # So we don't need to handle time gracefully inside, but we should be fast.
                
            return blocks_used

    # Run multiple greedy passes with different seeds and keep the best
    best_blocks = None
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    seed = 0
    while time.time() - start_time < timeout:
        blocks = greedy_run(seed)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
        seed += 1
        
    return best_blocks

def _generic_greedy(v, k, t):
    # Fallback for other parameters
    uncovered = set(itertools.combinations(range(v), t))
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
