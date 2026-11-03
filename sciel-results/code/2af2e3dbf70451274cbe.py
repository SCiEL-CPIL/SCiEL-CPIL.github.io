"""Improved covering design C(19,7,4) using bitset-based randomized greedy with redundancy removal."""
import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_to_subsets = [[] for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for p in subset:
            point_to_subsets[p].append(idx)
    
    # Bitset for uncovered t-subsets
    # Using Python integers as bitsets
    uncovered_mask = (1 << num_t) - 1  # all bits set
    
    def get_uncovered_count(mask):
        return bin(mask).count('1')
    
    def greedy_cover(uncovered, seed):
        """Run greedy covering with a given random seed for tie-breaking."""
        rng = random.Random(seed)
        blocks = []
        remaining = uncovered.copy()
        
        while remaining != 0:
            if time.time() > deadline:
                return blocks
            
            # Find the t-subset with the fewest candidates (or random tie-breaking)
            # Pick a random uncovered t-subset to start the block
            # Use random selection among uncovered subsets
            uncovered_list = [i for i in range(num_t) if (remaining >> i) & 1]
            if not uncovered_list:
                break
            
            # Pick a random uncovered t-subset
            target_idx = rng.choice(uncovered_list)
            target_subset = all_t_subsets[target_idx]
            
            # Build block: start with target_subset, then add points greedily
            block = list(target_subset)
            block_set = set(block)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_point = -1
                best_gain = -1
                best_count = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x forms an uncovered t-subset
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in t_subset_index:
                            idx = t_subset_index[new_subset]
                            if (remaining >> idx) & 1:
                                gain += 1
                    
                    if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                        best_gain = gain
                        best_point = x
                        best_count = gain
                
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update remaining mask: remove all t-subsets of the new block
                for ts in itertools.combinations(block, t):
                    if ts in t_subset_index:
                        idx = t_subset_index[ts]
                        remaining &= ~(1 << idx)
            
            block.sort()
            blocks.append(block)
        
        return blocks
    
    # Run multiple random greedy passes and keep the best
    best_blocks = None
    best_len = float('inf')
    
    num_trials = 20
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(uncovered_mask, seed=trial)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Check if we've reached the known lower bound or best known
        if best_len <= 151:
            break
    
    # Post-hoc redundancy removal: try to remove blocks that are redundant
    if best_blocks is not None:
        # Verify coverage and remove redundant blocks
        covered_mask = 0
        for block in best_blocks:
            for ts in itertools.combinations(block, t):
                if ts in t_subset_index:
                    idx = t_subset_index[ts]
                    covered_mask |= (1 << idx)
        
        # Check if all covered
        if covered_mask == uncovered_mask:
            # Try to remove redundant blocks
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(best_blocks) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    
                    # Temporarily remove block i
                    test_covered = 0
                    for j, block in enumerate(best_blocks):
                        if j == i:
                            continue
                        for ts in itertools.combinations(block, t):
                            if ts in t_subset_index:
                                idx = t_subset_index[ts]
                                test_covered |= (1 << idx)
                    
                    if test_covered == uncovered_mask:
                        best_blocks.pop(i)
                        changed = True
                        break
    
    # Final verification
    if best_blocks is None:
        # Fallback: return empty or minimal
        return []
    
    # Verify final coverage
    final_covered = 0
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            if ts in t_subset_index:
                idx = t_subset_index[ts]
                final_covered |= (1 << idx)
    
    if final_covered != uncovered_mask:
        # Coverage incomplete, fall back to full greedy
        blocks = greedy_cover(uncovered_mask, seed=0)
        return blocks
    
    return best_blocks
