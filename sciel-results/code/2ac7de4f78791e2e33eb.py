import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_coverage[i] = list of t_subset indices covered by block i
    block_coverage = []
    # subset_to_blocks[j] = list of block indices covering t_subset j
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            subset_to_blocks[idx].append(b_idx)
        block_coverage.append(covered_indices)
        
    # Helper: Check if a block is redundant given counts
    def is_redundant(b_idx, counts):
        for idx in block_coverage[b_idx]:
            if counts[idx] == 1:
                return False
        return True

    # Helper: Local Search to minimize size
    def local_search(blocks_list):
        # Initialize counts
        counts = [0] * num_t_subsets
        for b_idx in blocks_list:
            for idx in block_coverage[b_idx]:
                counts[idx] += 1
        
        # Phase 1: Reduction
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list)):
                b_idx = blocks_list[i]
                if is_redundant(b_idx, counts):
                    for idx in block_coverage[b_idx]:
                        counts[idx] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break # Restart as indices changed
                    
        # Phase 2: Swaps
        # We try to swap blocks to enable further reductions or direct size reduction?
        # Actually, a swap doesn't reduce size directly unless it enables a reduction.
        # So: Pick a block, find a replacement, swap, then run reduction.
        
        max_iters = 50 # Limit swap iterations per LS call to avoid getting stuck
        for _ in range(max_iters):
            if not blocks_list:
                break
            i = random.randint(0, len(blocks_list) - 1)
            old_b_idx = blocks_list[i]
            
            # Find unique subsets of old_b_idx
            unique_indices = [idx for idx in block_coverage[old_b_idx] if counts[idx] == 1]
            
            if not unique_indices:
                # Should have been removed in reduction, but maybe counts changed?
                # Just skip
                continue
            
            # Find a candidate block that covers all unique_indices
            # Pick the most constrained unique subset (fewest covering blocks)
            # Sort unique_indices by len(subset_to_blocks[idx])
            unique_indices.sort(key=lambda x: len(subset_to_blocks[x]))
            first_unique = unique_indices[0]
            
            candidates = subset_to_blocks[first_unique]
            found = False
            # Shuffle candidates to avoid bias
            random.shuffle(candidates)
            
            for cand in candidates:
                if cand == old_b_idx:
                    continue
                # Check if cand covers all unique_indices
                valid = True
                # Optimization: check other unique indices
                for u_idx in unique_indices[1:]:
                    if u_idx not in block_coverage[cand]:
                        valid = False
                        break
                if valid:
                    # Perform Swap
                    # Remove old
                    for idx in block_coverage[old_b_idx]:
                        counts[idx] -= 1
                    # Add new
                    for idx in block_coverage[cand]:
                        counts[idx] += 1
                    blocks_list[i] = cand
                    found = True
                    break
            
            if found:
                # Run reduction again
                changed = True
                while changed:
                    changed = False
                    for j in range(len(blocks_list)):
                        b_idx = blocks_list[j]
                        if is_redundant(b_idx, counts):
                            for idx in block_coverage[b_idx]:
                                counts[idx] -= 1
                            blocks_list.pop(j)
                            changed = True
                            break
            else:
                # No swap found for this block, try another random block
                continue
                
        return blocks_list, counts

    # Helper: Randomized Greedy Construction
    def greedy_construct():
        counts = [0] * num_t_subsets
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Iterate all blocks
            for b_idx in range(num_blocks):
                gain = 0
                for idx in block_coverage[b_idx]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b_idx)
            
            if best_gain <= 0:
                break # No block covers any uncovered subset? Should not happen.
                
            # Randomly pick from best candidates
            chosen = random.choice(best_candidates)
            current_blocks.append(chosen)
            
            for idx in block_coverage[chosen]:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return current_blocks

    # Helper: Perturbation
    def perturb(blocks_list, counts):
        # Randomly swap 1-3 blocks with random blocks from the universe
        # This might break coverage, but local_search will repair it
        new_blocks = list(blocks_list)
        
        # If we add random blocks, we increase size. We want to explore.
        # Let's just swap existing blocks with random ones.
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            if not new_blocks:
                break
            i = random.randint(0, len(new_blocks) - 1)
            new_blocks[i] = random.randint(0, num_blocks - 1)
            
        return new_blocks

    best_solution = []
    best_size = float('inf')
    
    # Initial construction
    current_blocks = greedy_construct()
    current_blocks, _ = local_search(current_blocks)
    best_solution = current_blocks
    best_size = len(current_blocks)
    
    # ILS Loop
    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(best_solution, None) # counts not needed for perturb structure, just indices
        
        # Local Search on perturbed
        # Note: local_search recalculates counts internally
        improved_blocks, _ = local_search(perturbed)
        
        if len(improved_blocks) < best_size:
            best_size = len(improved_blocks)
            best_solution = improved_blocks
        elif len(improved_blocks) == best_size:
            # Keep it as a candidate for future perturbations if it's different?
            # For simplicity, only update if strictly better, or randomly accept if same size (simulated annealing style)
            if random.random() < 0.1:
                best_solution = improved_blocks

    return [list(all_blocks[i]) for i in best_solution]
