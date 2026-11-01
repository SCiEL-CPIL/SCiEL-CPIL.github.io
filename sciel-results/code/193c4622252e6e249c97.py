import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy + Local Search (Redundancy Removal)
    with multiple randomized restarts.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for fast array access in local search
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Estimate max restarts based on time
    # Each restart ~ 0.1s? Let's try up to 50 restarts.
    max_restarts = 50
    
    for seed in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        
        # 1. Greedy Construction
        # uncovered is a set of t-subset tuples
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random pivot from uncovered
            pivot = random.choice(list(uncovered))
            block = list(pivot)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                best_candidates = []
                
                # Try all points not in block
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets S
                    # such that S \ {x} is a subset of current block
                    gain = 0
                    # Iterate through (t-1) subsets of current block
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_point == -1:
                    # Should not happen
                    break
                    
                block.append(random.choice(best_candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            to_remove = set(itertools.combinations(block, t))
            uncovered.difference_update(to_remove)
        
        # 2. Local Search: Redundancy Removal
        # Build coverage counts
        coverage_count = [0] * num_t_subsets
        block_subsets_indices = [] # list of lists of indices for each block
        
        for b in blocks:
            indices = []
            for s in itertools.combinations(b, t):
                idx = subset_to_idx[s]
                coverage_count[idx] += 1
                indices.append(idx)
            block_subsets_indices.append(indices)
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # It is redundant if for all s in block_subsets_indices[i], coverage_count[s] > 1
                redundant = True
                for idx in block_subsets_indices[i]:
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for idx in block_subsets_indices[i]:
                        coverage_count[idx] -= 1
                    
                    blocks.pop(i)
                    block_subsets_indices.pop(i)
                    changed = True
                    break # Restart the loop as indices changed
        
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Optional: Early exit if we hit a very good number?
        # Lower bound is 264. Best known 335.
        # If we find < 335, it's a record improvement.
        
    if best_blocks is None:
        # Fallback: return empty or simple greedy
        return []
        
    return best_blocks
