import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for fast array access
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Greedy Initialization
    def greedy_cover():
        uncovered = set(range(num_t_subsets))
        coverage_count = [0] * num_t_subsets
        blocks = []
        
        while uncovered:
            best_block = None
            max_gain = -1
            
            # We need to find a k-subset that covers the most uncovered t-subsets.
            # Brute force over all k-subsets is C(24,7) = 346,104.
            # For each block, we check how many of its t-subsets are in 'uncovered'.
            # This is feasible for the first few blocks, but 'uncovered' shrinks.
            # Optimization: Only consider blocks that contain at least one uncovered t-subset?
            # Actually, we can just iterate through all possible blocks.
            
            # To speed up, we can pre-compute the t-subsets for every possible block.
            # But C(24,7) is large. Let's just generate them on the fly or use a smarter heuristic.
            # A smarter heuristic: Pick a random uncovered t-subset, then build a block around it.
            # But greedy "best choice" is better.
            
            # Let's use a randomized greedy to allow for multiple restarts/improvements later.
            # Or just standard greedy.
            
            # Standard Greedy:
            # Iterate through all k-subsets? 346k * (k choose t) operations.
            # (7 choose 3) = 35. 346k * 35 ~ 12 million ops per block.
            # If we need ~70 blocks, that's 840 million ops. Might be too slow in Python.
            
            # Faster Greedy:
            # 1. Pick a random uncovered t-subset S.
            # 2. The next block MUST contain S.
            # 3. We need to choose k-t points from the remaining v-t points.
            # 4. C(21, 4) = 5985 possibilities.
            # 5. For each candidate block (S + 4 other points), count how many NEW uncovered t-subsets it covers.
            # 6. Pick the best one.
            
            s = random.choice(list(uncovered))
            s_points = set(s)
            
            remaining_points = [p for p in range(v) if p not in s_points]
            best_gain = -1
            best_block = None
            
            # We need to choose k-t points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(s_points.union(combo)))
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(block, t):
                    idx = subset_to_idx[ts]
                    if idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    
            if best_block is None:
                # Fallback, shouldn't happen
                best_block = tuple(sorted(s_points.union(remaining_points[:k-t])))
                
            blocks.append(list(best_block))
            
            # Update coverage and uncovered
            for ts in itertools.combinations(best_block, t):
                idx = subset_to_idx[ts]
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks, coverage_count

    # Local Search: Add then Remove
    def local_search(blocks, coverage_count):
        uncovered = set()
        for i, c in enumerate(coverage_count):
            if c == 0:
                uncovered.add(i)
        
        if uncovered:
            # Greedy fix if somehow invalid (shouldn't happen)
            return blocks, coverage_count

        while time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant blocks
            for i in range(len(blocks)):
                # Check if block i is removable
                removable = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = subset_to_idx[ts]
                    if coverage_count[idx] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        idx = subset_to_idx[ts]
                        coverage_count[idx] -= 1
                        if coverage_count[idx] == 0:
                            uncovered.add(idx)
                    
                    blocks.pop(i)
                    improved = True
                    break # Restart the loop to re-evaluate removals
            
            if improved:
                continue
            
            # 2. If no removal, try a perturbation: Add a random block, then try to remove
            # This allows escaping local minima
            if random.random() < 0.5: # 50% chance to perturb
                # Generate a random block
                new_block = list(random.sample(range(v), k))
                new_block.sort()
                
                # Add it
                blocks.append(new_block)
                for ts in itertools.combinations(new_block, t):
                    idx = subset_to_idx[ts]
                    coverage_count[idx] += 1
                    if coverage_count[idx] == 1:
                        uncovered.discard(idx)
                
                # Now try to remove any block (including the new one)
                # We prefer removing old blocks to keep the count down or same
                # If we remove the new block, we revert.
                # If we remove an old block, we have same count but different structure.
                
                # Try to remove one block
                removed_any = False
                for j in range(len(blocks)):
                    removable = True
                    for ts in itertools.combinations(blocks[j], t):
                        idx = subset_to_idx[ts]
                        if coverage_count[idx] <= 1:
                            removable = False
                            break
                    
                    if removable:
                        # Remove block j
                        for ts in itertools.combinations(blocks[j], t):
                            idx = subset_to_idx[ts]
                            coverage_count[idx] -= 1
                            if coverage_count[idx] == 0:
                                uncovered.add(idx)
                        
                        blocks.pop(j)
                        removed_any = True
                        break
                        
                if not removed_any:
                    # Could not remove anything, revert the addition
                    # Remove the last added block (index -1)
                    last_block = blocks.pop()
                    for ts in itertools.combinations(last_block, t):
                        idx = subset_to_idx[ts]
                        coverage_count[idx] -= 1
                        if coverage_count[idx] == 0:
                            uncovered.add(idx)

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts of Greedy + Local Search
    while time.time() < deadline:
        blocks, coverage_count = greedy_cover()
        
        # Validate
        valid = True
        for i, c in enumerate(coverage_count):
            if c == 0:
                valid = False
                break
        
        if not valid:
            continue
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy
            
        # Local Search
        local_search(blocks, coverage_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # Early stop if we hit the lower bound or a very good value?
        # Lower bound is 69. Best known 76.
        if best_len <= 70:
            break

    return best_blocks
