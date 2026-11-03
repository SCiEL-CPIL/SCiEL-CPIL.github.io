import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin for return processing

    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage: list of lists of t-subset indices
    block_coverage = []
    for block in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(block, t):
            covered_indices.append(t_subset_to_idx[ts])
        block_coverage.append(covered_indices)
        
    # Precompute inverse index: t_subset_idx -> list of block indices covering it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverage):
        for idx in cov:
            t_to_blocks[idx].append(b_idx)

    def greedy_init(seed=0):
        random.seed(seed)
        counts = [0] * num_t_subsets
        current_blocks = []
        uncovered = set(range(num_t_subsets))
        
        # Greedy set cover
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            # Optimization: Only check blocks that cover at least one uncovered subset?
            # Hard to do efficiently without a heap or similar structure.
            # With 75k blocks and ~300 steps, 22M iterations is okay in Python if simple.
            # But let's try to speed it up.
            
            # We can maintain a list of "active" blocks? No, too complex.
            # Let's just iterate.
            for b_idx in range(num_blocks):
                gain = 0
                for idx in block_coverage[b_idx]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Early termination if gain is max possible? Max is C(k,t)
                    if best_gain == 56: # C(8,5) for this specific case, but generalize?
                         # Actually, max gain is min(len(uncovered), C(k,t))
                         # We can break if gain == len(uncovered) or gain == C(k,t)
                         pass
            
            if best_block_idx == -1:
                break
                
            current_blocks.append(best_block_idx)
            for idx in block_coverage[best_block_idx]:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return current_blocks, counts

    def reduce_blocks(blocks_list, counts):
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
        return blocks_list, counts

    best_blocks = []
    best_size = float('inf')
    
    # Run multiple restarts with different seeds
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb the greedy? Or just different random seed for tie-breaking?
        # The current greedy is deterministic. To get diversity, we can shuffle the block order?
        # Or pick randomly among top K gainers.
        
        # Let's implement a randomized greedy for diversity
        counts = [0] * num_t_subsets
        current_blocks = []
        uncovered = set(range(num_t_subsets))
        rng = random.Random(r)
        
        while uncovered:
            # Find top candidates
            gains = [0] * num_blocks
            for b_idx in range(num_blocks):
                gain = 0
                for idx in block_coverage[b_idx]:
                    if idx in uncovered:
                        gain += 1
                gains[b_idx] = gain
            
            max_gain = max(gains)
            if max_gain == 0:
                break
                
            # Pick randomly from blocks with gain == max_gain
            # Or pick from top 10%?
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            best_block_idx = rng.choice(candidates)
            
            current_blocks.append(best_block_idx)
            for idx in block_coverage[best_block_idx]:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
            
        # Reduce
        current_blocks, counts = reduce_blocks(current_blocks, counts)
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # Local Search on the reduced solution
        # We need to keep counts consistent
        # If we modified best_blocks, we should recompute counts for it if we want to do local search on it.
        # But local search modifies the solution.
        
        # Let's do local search on the current solution (which is already reduced)
        # We need counts to be correct for current_blocks.
        # reduce_blocks updates counts correctly.
        
        stuck_counter = 0
        local_start = time.time()
        
        while time.time() - start_time < time_limit and (time.time() - local_start) < 5.0:
            # Try to reduce again (in case previous swaps created redundancies)
            new_blocks, new_counts = reduce_blocks(current_blocks, counts)
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                counts = new_counts
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                stuck_counter = 0
            else:
                stuck_counter += 1
                
            if stuck_counter > 10:
                # Try a swap
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    old_b_idx = current_blocks[i]
                    
                    # Find unique subsets of old_b_idx
                    unique_indices = []
                    for idx in block_coverage[old_b_idx]:
                        if counts[idx] == 1:
                            unique_indices.append(idx)
                    
                    if unique_indices:
                        # Find candidates covering all unique_indices
                        # Use inverse index for the first unique subset
                        first_unique = unique_indices[0]
                        candidates = t_to_blocks[first_unique]
                        
                        valid_candidates = []
                        for cand in candidates:
                            if cand == old_b_idx:
                                continue
                            # Check if cand covers all unique_indices
                            # Optimization: check if set(unique_indices) is subset of set(block_coverage[cand])
                            # Using sets for this specific check is fast enough for small lists
                            cand_set = set(block_coverage[cand])
                            if all(u in cand_set for u in unique_indices):
                                valid_candidates.append(cand)
                                
                        if valid_candidates:
                            new_b_idx = rng.choice(valid_candidates)
                            
                            # Perform swap
                            for idx in block_coverage[old_b_idx]:
                                counts[idx] -= 1
                            for idx in block_coverage[new_b_idx]:
                                counts[idx] += 1
                            current_blocks[i] = new_b_idx
                            
                            # Reset stuck counter to allow more swaps
                            stuck_counter = 0
                            
    # Return the best blocks found
    return [list(all_blocks[i]) for i in best_blocks]
