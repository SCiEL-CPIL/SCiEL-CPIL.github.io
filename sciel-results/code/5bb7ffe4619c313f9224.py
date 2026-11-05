import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        # Initialize coverage counts
        # Using a dict for sparse counts
        cov_count = {sub: 0 for sub in all_t_subsets}
        uncovered_count = len(all_t_subsets)
        
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered subset
            # To pick randomly from uncovered, we can find one with count 0
            # This is O(N) scan. N=3876. 150 blocks * 3876 = ~600k ops. Fine.
            target_subset = None
            # Randomize the start of the scan to pick different ones
            start_idx = random.randint(0, len(all_t_subsets) - 1)
            for i in range(len(all_t_subsets)):
                idx = (start_idx + i) % len(all_t_subsets)
                if cov_count[all_t_subsets[idx]] == 0:
                    target_subset = all_t_subsets[idx]
                    break
            
            if target_subset is None:
                break # Should not happen if uncovered_count > 0
                
            # Start block with target_subset
            block = list(target_subset)
            
            # Add points to reach size k
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check t-1 combinations of current block
                    for comb in itertools.combinations(block, t-1):
                        new_sub = tuple(sorted(comb + (x,)))
                        if cov_count[new_sub] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    block.append(random.choice(best_points))
                else:
                    # Fallback
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for sub in itertools.combinations(block, t):
                if cov_count[sub] == 0:
                    uncovered_count -= 1
                cov_count[sub] += 1
        
        # Post-processing: Remove redundant blocks
        final_blocks = []
        for i, b in enumerate(blocks):
            redundant = True
            for sub in itertools.combinations(b, t):
                if cov_count[sub] == 1:
                    redundant = False
                    break
            if not redundant:
                final_blocks.append(b)
        
        if len(final_blocks) < best_count:
            best_count = len(final_blocks)
            best_blocks = final_blocks
            
        if best_count <= 131:
            break

    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
