import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage
    block_coverage = []
    block_coverage_sets = []
    for block in all_blocks:
        covered_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverage.append(covered_indices)
        block_coverage_sets.append(set(covered_indices))
        
    # Precompute inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverage):
        for idx in cov:
            t_to_blocks[idx].append(b_idx)
            
    # 3. Greedy Initialization
    counts = [0] * num_t_subsets
    current_blocks = []
    
    uncovered_count = num_t_subsets
    
    while uncovered_count > 0:
        best_block_idx = -1
        best_gain = 0
        
        for b_idx in range(num_blocks):
            gain = 0
            # Use set intersection? 
            # gain = len(block_coverage_sets[b_idx] & uncovered_set)?
            # We don't have an uncovered_set, we have counts.
            # So we must loop.
            for idx in block_coverage[b_idx]:
                if counts[idx] == 0:
                    gain += 1
                    if gain > best_gain:
                        if gain == len(block_coverage[b_idx]):
                            break
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                if best_gain == len(block_coverage[b_idx]):
                    break
        
        if best_block_idx == -1:
            break
            
        current_blocks.append(best_block_idx)
        for idx in block_coverage[best_block_idx]:
            if counts[idx] == 0:
                uncovered_count -= 1
            counts[idx] += 1
                
    # 4. Local Search: Reduction and Swap
    
    def try_reduce(blocks_list):
        """Try to remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list)):
                b_idx = blocks_list[i]
                is_redundant = True
                for idx in block_coverage[b_idx]:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for idx in block_coverage[b_idx]:
                        counts[idx] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    current_blocks = try_reduce(current_blocks)
    
    if time.time() - start_time > time_limit * 0.4:
        return [list(all_blocks[i]) for i in current_blocks]
        
    best_size = len(current_blocks)
    best_blocks_copy = current_blocks[:]
    
    while time.time() - start_time < time_limit:
        if len(current_blocks) > 0:
            i = random.randint(0, len(current_blocks) - 1)
            old_b_idx = current_blocks[i]
            
            unique_indices = []
            for idx in block_coverage[old_b_idx]:
                if counts[idx] == 1:
                    unique_indices.append(idx)
            
            if not unique_indices:
                # Redundant
                for idx in block_coverage[old_b_idx]:
                    counts[idx] -= 1
                current_blocks.pop(i)
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks_copy = current_blocks[:]
                continue

            # Find most constrained unique index
            min_cand_idx = min(unique_indices, key=lambda x: len(t_to_blocks[x]))
            candidates = t_to_blocks[min_cand_idx]
            
            valid_candidates = []
            for cand in candidates:
                if cand == old_b_idx:
                    continue
                if all(u_idx in block_coverage_sets[cand] for u_idx in unique_indices):
                    valid_candidates.append(cand)
            
            if valid_candidates:
                # Save state
                old_blocks_state = current_blocks[:]
                old_counts_state = counts[:]
                
                # Try a random candidate
                new_b_idx = random.choice(valid_candidates)
                
                # Perform swap
                for idx in block_coverage[old_b_idx]:
                    counts[idx] -= 1
                for idx in block_coverage[new_b_idx]:
                    counts[idx] += 1
                current_blocks[i] = new_b_idx
                
                # Reduce
                current_blocks = try_reduce(current_blocks)
                
                if len(current_blocks) <= best_size:
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks_copy = current_blocks[:]
                    # Keep state (improvement or equal)
                else:
                    # Revert
                    current_blocks = old_blocks_state
                    counts = old_counts_state

    return [list(all_blocks[i]) for i in best_blocks_copy]
