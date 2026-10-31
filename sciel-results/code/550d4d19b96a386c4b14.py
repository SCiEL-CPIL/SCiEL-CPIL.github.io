import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast counting
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all possible k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(points, k))
    
    # Pre-compute for each k-subset, the list of t-subsets it covers
    # block_t_mask[block_idx] = list of t_subset indices
    block_t_indices = []
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            covered_ts.append(t_subset_index[ts])
        block_t_indices.append(covered_ts)
        
    # Greedy Phase
    coverage_counts = [0] * n_t_subsets
    selected_blocks = []
    
    while True:
        # Check if all covered
        if all(c > 0 for c in coverage_counts):
            break
            
        # Find best block
        best_block_idx = -1
        best_gain = -1
        
        # We can optimize this by only considering blocks that cover at least one uncovered subset
        # But with 50k blocks, a full scan is fast enough in Python if we use vectorization or simple loops
        # Let's use a simple loop first, it's O(N_blocks * t_subsets_per_block)
        # 50k * ~50 = 2.5M operations per iteration. ~150 iterations -> 375M ops. Might be too slow in pure Python.
        
        # Optimization: Use numpy for coverage updates? Or just optimize the inner loop.
        # Let's try a slightly smarter greedy:
        # Pick a random uncovered t-subset, and pick the block that covers it and maximizes total uncovered coverage.
        
        uncovered_ts = [i for i, c in enumerate(coverage_counts) if c == 0]
        if not uncovered_ts:
            break
            
        # Pick a target uncovered subset
        target_ts = random.choice(uncovered_ts)
        
        # Find all blocks that cover target_ts
        candidates = [b_idx for b_idx, ts_list in enumerate(block_t_indices) if target_ts in ts_list]
        
        # Among candidates, pick the one that covers the most uncovered subsets
        best_candidate = -1
        max_uncovered_covered = -1
        
        for b_idx in candidates:
            count_uncovered = 0
            for ts_idx in block_t_indices[b_idx]:
                if coverage_counts[ts_idx] == 0:
                    count_uncovered += 1
            if count_uncovered > max_uncovered_covered:
                max_uncovered_covered = count_uncovered
                best_candidate = b_idx
                
        if best_candidate == -1:
            break
            
        selected_blocks.append(best_candidate)
        for ts_idx in block_t_indices[best_candidate]:
            coverage_counts[ts_idx] += 1
            
        if time.time() - start_time > time_limit * 0.2:
            break

    # Convert indices back to blocks
    current_blocks = [all_k_subsets[i] for i in selected_blocks]
    
    # Local Search Phase
    # State: list of block indices
    # Objective: Minimize len(blocks)
    
    # Re-initialize coverage for local search
    coverage_counts = [0] * n_t_subsets
    for b_idx in selected_blocks:
        for ts_idx in block_t_indices[b_idx]:
            coverage_counts[ts_idx] += 1
            
    # Helper to check if a block can be removed
    def can_remove(b_idx):
        for ts_idx in block_t_indices[b_idx]:
            if coverage_counts[ts_idx] == 1:
                return False
        return True
        
    # Helper to apply removal
    def remove_block(b_idx):
        for ts_idx in block_t_indices[b_idx]:
            coverage_counts[ts_idx] -= 1
            
    # Helper to apply addition
    def add_block(b_idx):
        for ts_idx in block_t_indices[b_idx]:
            coverage_counts[ts_idx] += 1
            
    # Simulated Annealing / Hill Climbing
    # We will try to swap blocks or remove blocks.
    
    # Pre-compute for each block, the set of t-subsets it covers as a set for fast intersection?
    # block_t_sets = [set(ts_list) for ts_list in block_t_indices]
    
    # To speed up swap logic:
    # A block B_old can be replaced by B_new if:
    # 1. All t-subsets covered only by B_old (in current solution) are also covered by B_new.
    # 2. This implies coverage_counts[ts] >= 2 for all ts in B_old \ B_new? No.
    #    It implies that for any ts in B_old, if coverage_counts[ts] == 1, then ts must be in B_new.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try removing any block
        for i in range(len(selected_blocks)):
            b_idx = selected_blocks[i]
            if can_remove(b_idx):
                remove_block(b_idx)
                selected_blocks.pop(i)
                improved = True
                break # Restart loop after modification
                
        if improved:
            continue
            
        # 2. Try swapping
        # Pick a random block to replace
        if selected_blocks:
            i = random.randrange(len(selected_blocks))
            b_old = selected_blocks[i]
            
            # Find t-subsets that are uniquely covered by b_old
            unique_ts = [ts_idx for ts_idx in block_t_indices[b_old] if coverage_counts[ts_idx] == 1]
            
            if not unique_ts:
                # This block is redundant? Should have been caught by removal step.
                # But maybe it's not strictly redundant in terms of count, but logically?
                # If no unique ts, it can be removed.
                remove_block(b_old)
                selected_blocks.pop(i)
                improved = True
                continue
                
            # Find a candidate b_new that covers all unique_ts
            # And preferably covers many other uncovered or low-count ts
            best_new = -1
            best_score = -1
            
            # Sample candidates to save time? Or iterate all?
            # Iterating all 50k blocks is expensive if done for every swap attempt.
            # Let's filter candidates: must cover all unique_ts.
            
            # Optimization: Pre-compute for each t_subset, the list of blocks that cover it.
            # ts_to_blocks = {ts_idx: [] for ts_idx in range(n_t_subsets)}
            # for b_idx, ts_list in enumerate(block_t_indices):
            #     for ts_idx in ts_list:
            #         ts_to_blocks[ts_idx].append(b_idx)
            
            # This takes memory but speeds up candidate finding.
            # Let's build it once.
            if not hasattr(construct, 'ts_to_blocks'):
                construct.ts_to_blocks = [[] for _ in range(n_t_subsets)]
                for b_idx, ts_list in enumerate(block_t_indices):
                    for ts_idx in ts_list:
                        construct.ts_to_blocks[ts_idx].append(b_idx)
            
            # Intersection of lists of blocks covering each unique_ts
            candidate_set = set(construct.ts_to_blocks[unique_ts[0]])
            for ts in unique_ts[1:]:
                candidate_set.intersection_update(construct.ts_to_blocks[ts])
                
            if not candidate_set:
                continue
                
            # Pick best candidate from candidate_set
            # Score: number of ts with count 1 that it covers (besides the required ones) + random tie break
            # Actually, we just need it to be valid. To improve diversity, pick randomly or by potential.
            # Let's pick the one that covers the most total ts with count == 1 (to create redundancy for future removals)
            
            for b_new in candidate_set:
                if b_new == b_old:
                    continue
                # Check if b_new is already in selected_blocks?
                # Swapping with an existing block is just a deletion of one and addition of another?
                # No, if b_new is already there, we are just removing b_old.
                # That case is handled by removal if valid.
                
                # Score: how many "critical" (count==1) ts does it cover?
                score = sum(1 for ts_idx in block_t_indices[b_new] if coverage_counts[ts_idx] == 1)
                
                if score > best_score:
                    best_score = score
                    best_new = b_new
                    
            if best_new != -1:
                # Perform swap
                remove_block(b_old)
                add_block(best_new)
                selected_blocks[i] = best_new
                improved = True
                # Don't break, try more swaps? Or break to re-evaluate removals?
                # Breaking allows the outer loop to check for removals first, which is safer.
                break
                
    # Convert back to list of lists
    final_blocks = [list(all_k_subsets[i]) for i in selected_blocks]
    return final_blocks
