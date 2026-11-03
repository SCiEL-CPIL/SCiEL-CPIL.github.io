import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Pre-compute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, pre-compute which t-subsets it covers
    # block_covered[i] = list of t-subset indices covered by block i
    block_covered = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_subset_to_idx[ts])
        block_covered.append(covered)
    
    # Primal-dual algorithm
    def primal_dual_cover():
        # dual[i] = dual variable for t-subset i
        dual = [0.0] * total_t_subsets
        # uncovered[i] = True if t-subset i is not yet covered
        uncovered = [True] * total_t_subsets
        uncovered_count = total_t_subsets
        selected_blocks = []
        
        while uncovered_count > 0:
            # Find uncovered t-subsets
            uncovered_indices = [i for i in range(total_t_subsets) if uncovered[i]]
            
            if not uncovered_indices:
                break
            
            # Raise dual variables for uncovered t-subsets
            # We raise them until some block becomes "tight" (sum of duals = 1)
            # For efficiency, we use a heuristic: raise all uncovered equally
            
            # Find the block that would become tight first
            # For each block, compute how much we can raise duals before it becomes tight
            # Block j is tight when sum_{i in block_covered[j]} dual[i] = 1
            
            best_block = -1
            best_time = float('inf')
            
            for j in range(num_blocks):
                # Check if this block covers any uncovered t-subsets
                has_uncovered = False
                current_sum = 0.0
                for i in block_covered[j]:
                    if uncovered[i]:
                        has_uncovered = True
                        current_sum += dual[i]
                    else:
                        current_sum += dual[i]
                
                if not has_uncovered:
                    continue
                
                # Time to reach tightness: (1 - current_sum) / count_uncovered_in_block
                count_uncovered = sum(1 for i in block_covered[j] if uncovered[i])
                if count_uncovered > 0:
                    time_to_tight = (1.0 - current_sum) / count_uncovered
                    if time_to_tight < best_time:
                        best_time = time_to_tight
                        best_block = j
            
            if best_block == -1:
                # Fallback: pick a random uncovered t-subset and find a block covering it
                pivot_idx = random.choice(uncovered_indices)
                pivot_ts = all_t_subsets[pivot_idx]
                pivot_set = set(pivot_ts)
                
                # Find all blocks containing pivot
                best_gain = -1
                for j in range(num_blocks):
                    if all(p in set(all_blocks[j]) for p in pivot_set):
                        gain = sum(1 for i in block_covered[j] if uncovered[i])
                        if gain > best_gain:
                            best_gain = gain
                            best_block = j
                
                if best_block == -1:
                    # Should not happen, but safety
                    break
                
                # Select this block
                selected_blocks.append(list(all_blocks[best_block]))
                for i in block_covered[best_block]:
                    if uncovered[i]:
                        uncovered[i] = False
                        uncovered_count -= 1
                continue
            
            # Raise duals for all uncovered t-subsets by best_time
            for i in uncovered_indices:
                dual[i] += best_time
            
            # Select the tight block
            selected_blocks.append(list(all_blocks[best_block]))
            for i in block_covered[best_block]:
                if uncovered[i]:
                    uncovered[i] = False
                    uncovered_count -= 1
        
        return selected_blocks
    
    # Greedy with dual weighting (faster alternative)
    def greedy_dual_weighted():
        uncovered = [True] * total_t_subsets
        uncovered_count = total_t_subsets
        # Weight for each t-subset: inversely proportional to how many blocks cover it
        # Pre-compute coverage frequency
        cover_freq = [0] * total_t_subsets
        for j in range(num_blocks):
            for i in block_covered[j]:
                cover_freq[i] += 1
        
        # Weight: 1/sqrt(cover_freq) to prioritize rare t-subsets
        weights = [1.0 / math.sqrt(cover_freq[i] + 1) for i in range(total_t_subsets)]
        
        blocks = []
        while uncovered_count > 0:
            # Find the block with maximum weighted gain
            best_block = -1
            best_weighted_gain = -1.0
            
            for j in range(num_blocks):
                weighted_gain = 0.0
                for i in block_covered[j]:
                    if uncovered[i]:
                        weighted_gain += weights[i]
                
                if weighted_gain > best_weighted_gain:
                    best_weighted_gain = weighted_gain
                    best_block = j
            
            if best_block == -1:
                break
            
            blocks.append(list(all_blocks[best_block]))
            for i in block_covered[best_block]:
                if uncovered[i]:
                    uncovered[i] = False
                    uncovered_count -= 1
            
            # Update weights: increase weight for newly uncovered rare t-subsets
            # Actually, keep weights fixed for simplicity
        
        return blocks
    
    # Local search: try to remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Build coverage map: for each t-subset, which blocks cover it
        t_to_blocks = {i: [] for i in range(total_t_subsets)}
        for idx, block in enumerate(blocks):
            block_set = set(block)
            for i in range(total_t_subsets):
                ts = all_t_subsets[i]
                if all(p in block_set for p in ts):
                    t_to_blocks[i].append(idx)
        
        # Try to remove each block if it's redundant
        changed = True
        while changed:
            changed = False
            for idx in range(len(blocks)):
                # Check if removing block idx still covers everything
                can_remove = True
                for i in range(total_t_subsets):
                    if len(t_to_blocks[i]) == 1 and t_to_blocks[i][0] == idx:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block idx
                    for i in range(total_t_subsets):
                        if idx in t_to_blocks[i]:
                            t_to_blocks[i].remove(idx)
                    blocks.pop(idx)
                    changed = True
                    break
        
        return blocks
    
    # Run multiple strategies
    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Primal-dual
    if time.time() < deadline:
        blocks = primal_dual_cover()
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Strategy 2: Greedy with dual weighting
    if time.time() < deadline:
        blocks = greedy_dual_weighted()
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Strategy 3: Multiple random restarts of greedy
    random.seed(42)
    for _ in range(20):
        if time.time() >= deadline - 5:
            break
        
        # Randomized greedy
        uncovered = [True] * total_t_subsets
        uncovered_count = total_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset as pivot
            uncovered_indices = [i for i in range(total_t_subsets) if uncovered[i]]
            if not uncovered_indices:
                break
            pivot_idx = random.choice(uncovered_indices)
            pivot_ts = all_t_subsets[pivot_idx]
            pivot_set = set(pivot_ts)
            
            # Find blocks containing pivot, pick one with max gain
            best_gain = -1
            best_block = -1
            for j in range(num_blocks):
                if all(p in set(all_blocks[j]) for p in pivot_set):
                    gain = sum(1 for i in block_covered[j] if uncovered[i])
                    if gain > best_gain:
                        best_gain = gain
                        best_block = j
            
            if best_block == -1:
                break
            
            blocks.append(list(all_blocks[best_block]))
            for i in block_covered[best_block]:
                if uncovered[i]:
                    uncovered[i] = False
                    uncovered_count -= 1
        
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = [True] * total_t_subsets
        uncovered_count = total_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            # Find the block with maximum gain
            best_gain = -1
            best_block = -1
            for j in range(num_blocks):
                gain = sum(1 for i in block_covered[j] if uncovered[i])
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
            
            if best_block == -1:
                break
            
            blocks.append(list(all_blocks[best_block]))
            for i in block_covered[best_block]:
                if uncovered[i]:
                    uncovered[i] = False
                    uncovered_count -= 1
        
        best_blocks = remove_redundant(blocks)
    
    return best_blocks
