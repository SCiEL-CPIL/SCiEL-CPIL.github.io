import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)
    
    best_blocks = None
    best_count = float('inf')
    
    def _remove_redundant(blocks, uncovered=None, time_limit=None):
        """Remove redundant blocks iteratively."""
        if time_limit is None:
            time_limit = deadline - 1.0
        changed = True
        while changed:
            changed = False
            if time.time() > time_limit:
                break
                
            # Build coverage count
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[t_sub_to_idx[sub]] += 1
            
            # Check each block for redundancy
            redundant_indices = []
            for i, b in enumerate(blocks):
                is_essential = False
                for sub in itertools.combinations(b, t):
                    if coverage_count[t_sub_to_idx[sub]] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_indices.append(i)
                        
            if redundant_indices:
                idx = random.choice(redundant_indices)
                blocks.pop(idx)
                changed = True
        return blocks
    
    def _greedy_construct(seed_subset, uncovered):
        """Construct a block starting from seed_subset."""
        block = list(seed_subset)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            tied_candidates = []
            
            block_set = set(block)
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for rest in t_minus_1_subsets:
                    T = tuple(sorted(rest + (x,)))
                    if T in t_sub_to_idx:
                        idx = t_sub_to_idx[T]
                        if uncovered[idx]:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    tied_candidates = [x]
                elif gain == best_gain:
                    tied_candidates.append(x)
                    
            if best_x == -1:
                break
                
            if len(tied_candidates) > 1:
                best_x = random.choice(tied_candidates)
                
            block.append(best_x)
            
        block.sort()
        return block
    
    # Phase 1: Multiple greedy runs with randomization
    for run_id in range(300):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(run_id * 7919 + 12345)
        
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices)
            seed_subset = all_t_subsets[seed_idx]
            block = _greedy_construct(seed_subset, uncovered)
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                uncovered[t_sub_to_idx[sub]] = False
        
        # Remove redundant blocks
        blocks = _remove_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    # Phase 2: Local search improvement on best solution
    if best_blocks is not None and time.time() < deadline - 20.0:
        current_blocks = [list(b) for b in best_blocks]
        
        for iter_id in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Try to find a block that can be replaced with a smaller set
            improved = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 5.0:
                    break
                    
                # Temporarily remove block i
                removed_block = current_blocks[i]
                current_blocks_without = current_blocks[:i] + current_blocks[i+1:]
                
                # Check if coverage is still complete
                coverage_count = np.zeros(num_t, dtype=np.int32)
                for b in current_blocks_without:
                    for sub in itertools.combinations(b, t):
                        coverage_count[t_sub_to_idx[sub]] += 1
                
                # Find uncovered t-subsets
                uncovered_mask = coverage_count == 0
                if not uncovered_mask.any():
                    # Block i is redundant, remove it
                    current_blocks = current_blocks_without
                    improved = True
                    break
                else:
                    # Try to cover the uncovered subsets with a new block
                    uncovered_indices = np.where(uncovered_mask)[0]
                    if len(uncovered_indices) > 0:
                        seed_idx = random.choice(uncovered_indices)
                        seed_subset = all_t_subsets[seed_idx]
                        uncovered_for_greedy = uncovered_mask.copy()
                        new_block = _greedy_construct(seed_subset, uncovered_for_greedy)
                        
                        # Check if new block covers more than what was lost
                        new_coverage = np.zeros(num_t, dtype=np.int32)
                        for b in current_blocks_without:
                            for sub in itertools.combinations(b, t):
                                new_coverage[t_sub_to_idx[sub]] += 1
                        for sub in itertools.combinations(new_block, t):
                            new_coverage[t_sub_to_idx[sub]] += 1
                        
                        if not (new_coverage == 0).any():
                            current_blocks = current_blocks_without + [new_block]
                            improved = True
                            break
            
            if not improved:
                # No improvement, try perturbation: randomly replace a block
                if len(current_blocks) > 0:
                    idx = random.randrange(len(current_blocks))
                    # Find a random t-subset from the removed block
                    block_tsubs_list = list(itertools.combinations(current_blocks[idx], t))
                    if block_tsubs_list:
                        random_sub = random.choice(block_tsubs_list)
                        uncovered_for_greedy = np.ones(num_t, dtype=bool)
                        new_block = _greedy_construct(random_sub, uncovered_for_greedy)
                        current_blocks[idx] = new_block
                        current_blocks = _remove_redundant(current_blocks)
                        
                        if len(current_blocks) < best_count:
                            best_count = len(current_blocks)
                            best_blocks = [list(b) for b in current_blocks]
    
    return best_blocks
