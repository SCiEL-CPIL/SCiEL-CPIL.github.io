"""Simulated annealing approach for covering design C(19,7,4) with optimized local search and greedy initialization."""
import itertools
import random
import math
import time
import numpy as np

# Use a fixed seed for reproducibility within a single run, but allow randomness
random.seed(42)
np.random.seed(42)

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Leave 10 seconds buffer
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and the t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices = []
    block_t_sets = []
    for b in all_blocks:
        t_idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_t_indices.append(t_idxs)
        block_t_sets.append(set(t_idxs))
        
    # Precompute t_subset_to_blocks for greedy initialization
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for bi, t_idxs in enumerate(block_t_indices):
        for ti in t_idxs:
            t_subset_to_blocks[ti].append(bi)
            
    # Precompute block_t_indices as numpy arrays for faster operations
    block_t_indices_np = [np.array(t_idxs, dtype=np.int32) for t_idxs in block_t_indices]

    # 1. Greedy Initialization (Max Gain)
    def greedy_init():
        uncovered = set(range(num_t_subsets))
        blocks = []
        # To speed up, we can use a heuristic: pick the block that covers the most uncovered items.
        # Instead of scanning all blocks every time (slow), we can use the "random target" method
        # but improve the selection of the best block for that target.
        # Or, we can just run a few iterations of "pick random uncovered, pick best block for it".
        
        # A faster greedy:
        # 1. Start with empty set.
        # 2. While uncovered:
        #    a. Find block b that maximizes |Cover(b) & uncovered|.
        #    b. Add b.
        
        # Optimized version:
        # We maintain a count of how many times each t-subset is covered by the *selected* blocks? No, that's for removal.
        # For addition, we want to maximize new coverage.
        
        # Let's use the randomized greedy but with a better tie-breaking or selection.
        # The current one picks a random uncovered target, then best block for it.
        # This is O(1) to pick target, O(Candidates) to pick block.
        # Candidates for a t-subset of size 4 in v=19, k=7:
        # Number of blocks containing a specific 4-subset:
        # Choose 3 more elements from 15 remaining: C(15,3) = 455.
        # So we check ~455 blocks. This is fast.
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered))
            
            # Pick the best block covering this target
            candidates = t_subset_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = 0
            
            for bi in candidates:
                # Count uncovered t-subsets in this block
                gain = 0
                for ti in block_t_indices[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    
            if best_block_idx == -1:
                break
                
            blocks.append(best_block_idx)
            
            # Update uncovered
            for ti in block_t_indices[best_block_idx]:
                if ti in uncovered:
                    uncovered.remove(ti)
                    
        return blocks

    # Run greedy a few times and keep the best
    best_greedy_blocks = None
    best_greedy_len = float('inf')
    for _ in range(20):
        gb = greedy_init()
        if len(gb) < best_greedy_len:
            best_greedy_len = len(gb)
            best_greedy_blocks = gb
            
    current_blocks = list(best_greedy_blocks)
    initial_greedy_blocks = list(current_blocks)
    
    # 2. Simulated Annealing
    # State: current_blocks (list of block indices)
    # Coverage: array of counts
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for bi in current_blocks:
        coverage[block_t_indices_np[bi]] += 1
        
    uncovered_count = int(np.sum(coverage == 0))
    current_cost = len(current_blocks) + 1000 * uncovered_count
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    temp = 50.0
    cooling = 0.995
    min_temp = 0.01
    
    # Local search: remove redundant blocks
    # Optimized: only check blocks that might be redundant.
    # A block is redundant if all its t-subsets have coverage >= 2.
    def local_search_remove_redundant(blocks, cov):
        changed = True
        while changed:
            changed = False
            # Iterate in reverse
            for i in range(len(blocks) - 1, -1, -1):
                bi = blocks[i]
                t_idxs = block_t_indices_np[bi]
                # Check if any t-subset has coverage == 1
                # If min(coverage[t_idxs]) >= 2, it's redundant
                if np.min(cov[t_idxs]) >= 2:
                    cov[t_idxs] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks
    
    # Initial local search
    current_blocks = local_search_remove_redundant(current_blocks, coverage)
    uncovered_count = int(np.sum(coverage == 0))
    current_cost = len(current_blocks) + 1000 * uncovered_count
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_blocks = list(current_blocks)
    
    while time.time() - start_time < timeout:
        if temp < min_temp:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                bi = current_blocks[idx]
                t_idxs = block_t_indices_np[bi]
                
                # Calculate change in uncovered count
                add_uncovered = int(np.sum(coverage[t_idxs] == 1))
                
                new_uncovered = uncovered_count + add_uncovered
                new_cost = (len(current_blocks) - 1) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks.pop(idx)
                    coverage[t_idxs] -= 1
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    
                    # Local search after removal
                    current_blocks = local_search_remove_redundant(current_blocks, coverage)
                    uncovered_count = int(np.sum(coverage == 0))
                    current_cost = len(current_blocks) + 1000 * uncovered_count
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        
        elif move_type < 0.7:
            # Add a block
            # Bias towards blocks that cover uncovered t-subsets if any exist
            if uncovered_count > 0:
                # Pick a random uncovered t-subset
                uncov_list = np.where(coverage == 0)[0]
                target_idx = uncov_list[random.randint(0, len(uncov_list)-1)]
                candidates = t_subset_to_blocks[target_idx]
                bi = candidates[random.randint(0, len(candidates)-1)]
            else:
                bi = random.randint(0, num_blocks - 1)
                
            t_idxs = block_t_indices_np[bi]
            
            # Calculate change in uncovered count
            reduce_uncovered = int(np.sum(coverage[t_idxs] == 0))
            
            new_uncovered = uncovered_count - reduce_uncovered
            new_cost = (len(current_blocks) + 1) + 1000 * new_uncovered
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # Apply move
                current_blocks.append(bi)
                coverage[t_idxs] += 1
                uncovered_count = new_uncovered
                current_cost = new_cost
                
                # Local search after addition
                current_blocks = local_search_remove_redundant(current_blocks, coverage)
                uncovered_count = int(np.sum(coverage == 0))
                current_cost = len(current_blocks) + 1000 * uncovered_count
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                    
        else:
            # Swap a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                old_bi = current_blocks[idx]
                new_bi = random.randint(0, num_blocks - 1)
                
                if old_bi == new_bi:
                    continue
                    
                old_ts = block_t_indices_np[old_bi]
                new_ts = block_t_indices_np[new_bi]
                
                # Efficient delta calculation using precomputed sets
                old_set = block_t_sets[old_bi]
                new_set = block_t_sets[new_bi]
                
                only_old = old_set - new_set
                only_new = new_set - old_set
                
                inc_uncovered = int(np.sum(coverage[list(only_old)] == 1)) if only_old else 0
                dec_uncovered = int(np.sum(coverage[list(only_new)] == 0)) if only_new else 0
                
                delta_uncovered = inc_uncovered - dec_uncovered
                new_uncovered = uncovered_count + delta_uncovered
                new_cost = len(current_blocks) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks[idx] = new_bi
                    coverage[old_ts] -= 1
                    coverage[new_ts] += 1
                    
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    
                    # Local search after swap
                    current_blocks = local_search_remove_redundant(current_blocks, coverage)
                    uncovered_count = int(np.sum(coverage == 0))
                    current_cost = len(current_blocks) + 1000 * uncovered_count
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        
        temp *= cooling
        
    # Verify and return
    # Ensure best_blocks is valid
    test_cov = np.zeros(num_t_subsets, dtype=np.int32)
    for bi in best_blocks:
        test_cov[block_t_indices_np[bi]] += 1
        
    if np.any(test_cov == 0):
        # Fallback to initial greedy
        best_blocks = initial_greedy_blocks

    result_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    return result_blocks
